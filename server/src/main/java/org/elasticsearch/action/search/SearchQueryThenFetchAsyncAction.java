/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.elasticsearch.action.search;

import org.apache.logging.log4j.Logger;
import org.apache.lucene.search.TopFieldDocs;
import org.elasticsearch.action.ActionListener;
import org.elasticsearch.cluster.ClusterState;
import org.elasticsearch.cluster.routing.GroupShardsIterator;
import org.elasticsearch.search.SearchPhaseResult;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.internal.AliasFilter;
import org.elasticsearch.search.internal.SearchContext;
import org.elasticsearch.search.internal.ShardSearchRequest;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.transport.Transport;

import java.util.Map;
import java.util.concurrent.Executor;
import java.util.function.BiFunction;

import static org.elasticsearch.action.search.SearchPhaseController.getTopDocsSize;

class SearchQueryThenFetchAsyncAction extends AbstractSearchAsyncAction<SearchPhaseResult> {

    private final SearchPhaseController searchPhaseController;
    private final SearchProgressListener progressListener;

    // informations to track the best bottom top doc globally.
    private final int topDocsSize;
    private final int trackTotalHitsUpTo;
    private volatile BottomSortValuesCollector bottomSortCollector;

    SearchQueryThenFetchAsyncAction(final Logger logger, final SearchTransportService searchTransportService,
                                    final BiFunction<String, String, Transport.Connection> nodeIdToConnection,
                                    final Map<String, AliasFilter> aliasFilter,
                                    final Map<String, Float> concreteIndexBoosts,
                                    final SearchPhaseController searchPhaseController, final Executor executor,
                                    final QueryPhaseResultConsumer resultConsumer, final SearchRequest request,
                                    final ActionListener<SearchResponse> listener,
                                    final GroupShardsIterator<SearchShardIterator> shardsIts,
                                    final TransportSearchAction.SearchTimeProvider timeProvider,
                                    ClusterState clusterState, SearchTask task, SearchResponse.Clusters clusters) {
        super("query", logger, searchTransportService, nodeIdToConnection, aliasFilter, concreteIndexBoosts,
                executor, request, listener, shardsIts, timeProvider, clusterState, task,
                resultConsumer, request.getMaxConcurrentShardRequests(), clusters);
        this.topDocsSize = getTopDocsSize(request);
        this.trackTotalHitsUpTo = request.resolveTrackTotalHitsUpTo();
        this.searchPhaseController = searchPhaseController;
        this.progressListener = task.getProgressListener();

        // register the release of the query consumer to free up the circuit breaker memory
        // at the end of the search
        addReleasable(resultConsumer);

        boolean hasFetchPhase = request.source() == null ? true : request.source().size() > 0;
        progressListener.notifyListShards(SearchProgressListener.buildSearchShards(this.shardsIts),
            SearchProgressListener.buildSearchShards(toSkipShardsIts), clusters, hasFetchPhase);
    }

    protected void executePhaseOnShard(final SearchShardIterator shardIt,
                                       final SearchShardTarget shard,
                                       final SearchActionListener<SearchPhaseResult> listener) {
        ShardSearchRequest request = rewriteShardSearchRequest(super.buildShardSearchRequest(shardIt, listener.requestIndex));
        
        /** 
         * 2020.12 송상욱
         * 1. 상품을 묶는 것은 자신의 인덱스(기준상품) 내에서만 묶으면 된다. 
         * 기준상품이 검색상품을 묶어야 하는지 찾아볼 필요가 없는 것이다. 
         * 하지만, ES의 collapse search는 집계를 목적으로 만들어졌기 때문에, Multi Search 를 할때 다른 인덱스까지 모두 찾아본다. 
         * 집계에서는 맞는 방법이지만, 우리 상황에서는 성능저하만 가져올 뿐이다.
         * 2. 묶음 대표 리스트 40개를 만들고 그 하위로 묶음 상품을 하나씩 검색하여 채워넣는 방식을 사용하며 문제가 없다. 
         * 다만, 묶음 수만큼 추가 검색이 이루어지므로, 결과 size 만큼 검색시간이 늘어나게 된다. 
         * 게다가 메인쿼리를 그대로 사용하면서 하위 묶음키를 필터로 걸어주기 때문에, 그 무게는 메인쿼리와 거의 동일하다. 
         * 그래서 쿼리의 부하를 줄이는 것이 관건인데, 검색상품의 경우는 사실상 묶음상품이 없기 때문에, 자기 자신만 묶이게 된다. 
         * 그러므로, bundleKey로만 검색하면 상품을 찾을수 있기 때문에, filter만 남기고 키워드 검색부분은 모두 제거하여 쿼리를 경량화 시킬수 있다.
         * 
         * 예를들어 4개의 인덱스에 collapse multi index 검색을 할때, 각 인덱스의 primary 샤드가 한개라고 가정하면
         * 1. bundleKey가 해당되는 인덱스에만 검색한다.
         * 2. 검색시 기준상품은 쿼리그대로, 검색상품은 경량화된 filter 쿼리로만 검색한다.
         * ES  순수 collapse search는 4x4 로 16번 검색이 이루어지지만, 개선한 코드로는 4번의 검색만 이루어진다.
         * 게다가 일부조건이 제거된 경량화 쿼리이므로 그 효과가 4x2=8 배 이상이 된다.
         * 
         * 참고사항: collapse 대표그룹을 만드는 부하도 작지는 않으며, multi index 갯수가 늘어남에 따라 소요시간이 크게 늘어난다.
         * 코드 자체는 org.apache.lucene.search.grouping.CollapsingTopDocsCollector 에 위치하는데, 루씬의 코드이기도 하고, 수정할만한 개선로직이 보이지는 않는다.
         * 그룹핑 부분은 튜닝없이 그대로 가야할듯하다.
         * 
         * 테스트결과: 
         * 환경: PC에서 인덱스 2개(40GB)를 collapse multi search.
         * 방법: Jmeter, CPU 100% 사용
         * - 기존  : TPS 3
         * - 개선후: TPS 30
         */

        String indexName = request.shardId().getIndex().getName();
        QueryBuilder query = request.source().query();
        if (query instanceof BoolQueryBuilder) {
            BoolQueryBuilder boolQuery = (BoolQueryBuilder) query;
            if (boolQuery.filter() != null && boolQuery.filter().size() > 0) {
                QueryBuilder filterQuery = boolQuery.filter().get(0);
                if (filterQuery instanceof MatchQueryBuilder) {
                    MatchQueryBuilder matchQuery = (MatchQueryBuilder) filterQuery;
                    if (matchQuery.fieldName().equals("bundleKey")) {
                        // 번들이라면.
                        String value = matchQuery.value().toString();
                        // bundle 값에 :이 존재한다면 인덱스 필터링을 사용한다. 없다면 모든 인덱스에 요청을 보낸다.
                        int pos = value.indexOf(':');
                        if (pos > 0) {
                            String prefix = value.substring(0, pos);
                            // indexName 이름에는 -a -b 가 붙어있을수 있으므로, prefix가 인덱스이름에 포함되는지를 검사한다.
                            if (indexName.startsWith(prefix)) {
                                if (indexName.startsWith("main")) {
                                    // case1) 기준상품 번들키로 -> 기준상품 검색
                                    // 기준상품은 쿼리를 그대로 사용.
                                } else {
                                    // case2) 검색상품 번들키로 -> 검색상품 검색
                                    //검색상품은 must 조건을 삭제하고 계속 검색.
                                    ((BoolQueryBuilder) query).must().clear();
                                }
                            } else {
                                // case3) 검색상품 번들키로 -> 기준상품 검색
                                // case4) 기준상품 번들키로 -> 검색상품 검색
                                // 인덱스가 다르다면..
                                // 동일 인덱스에만 검색을 해야하므로, early termination 적용.
                                listener.onResponse(QuerySearchResult.nullInstance());
                                return;
                            }
                        }
                    }
                }
            }
        }

        getSearchTransport().sendExecuteQuery(getConnection(shard.getClusterAlias(), shard.getNodeId()), request, getTask(), listener);
    }

    @Override
    protected void onShardGroupFailure(int shardIndex, SearchShardTarget shardTarget, Exception exc) {
        progressListener.notifyQueryFailure(shardIndex, shardTarget, exc);
    }

    @Override
    protected void onShardResult(SearchPhaseResult result, SearchShardIterator shardIt) {
        QuerySearchResult queryResult = result.queryResult();
        if (queryResult.isNull() == false
                // disable sort optims for scroll requests because they keep track of the last bottom doc locally (per shard)
                && getRequest().scroll() == null
                && queryResult.topDocs() != null
                && queryResult.topDocs().topDocs.getClass() == TopFieldDocs.class) {
            TopFieldDocs topDocs = (TopFieldDocs) queryResult.topDocs().topDocs;
            if (bottomSortCollector == null) {
                synchronized (this) {
                    if (bottomSortCollector == null) {
                        bottomSortCollector = new BottomSortValuesCollector(topDocsSize, topDocs.fields);
                    }
                }
            }
            bottomSortCollector.consumeTopDocs(topDocs, queryResult.sortValueFormats());
        }
        super.onShardResult(result, shardIt);
    }

    @Override
    protected SearchPhase getNextPhase(final SearchPhaseResults<SearchPhaseResult> results, SearchPhaseContext context) {
        return new FetchSearchPhase(results, searchPhaseController, null, this);
    }

    private ShardSearchRequest rewriteShardSearchRequest(ShardSearchRequest request) {
        if (bottomSortCollector == null) {
            return request;
        }

        // disable tracking total hits if we already reached the required estimation.
        if (trackTotalHitsUpTo != SearchContext.TRACK_TOTAL_HITS_ACCURATE
                && bottomSortCollector.getTotalHits() > trackTotalHitsUpTo) {
            request.source(request.source().shallowCopy().trackTotalHits(false));
        }

        // set the current best bottom field doc
        if (bottomSortCollector.getBottomSortValues() != null) {
            request.setBottomSortValues(bottomSortCollector.getBottomSortValues());
        }
        return request;
    }
}
