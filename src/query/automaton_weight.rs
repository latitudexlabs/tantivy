use crate::postings::TermInfo;
use crate::query::fuzzy_query::{DfaWrapper, IntersectionState, StartsWithAutomatonState};
use crate::query::score_combiner::SumCombiner;
use std::any::{Any, TypeId};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io;
use std::sync::Arc;

use common::BitSet;
use tantivy_fst::Automaton;

use super::fuzzy_query::StartsWithAutomaton;
use super::phrase_prefix_query::prefix_end;
use super::BitSetDocSet;
use crate::index::SegmentReader;
use crate::query::{ConstScorer, Explanation, Scorer, Weight};
use crate::schema::{Field, IndexRecordOption};
use crate::termdict::{TermDictionary, TermWithStateStreamer};
use crate::{DocId, Score, TantivyError};

/// A weight struct for Fuzzy Term and Regex Queries
pub struct AutomatonWeight<A> {
    field: Field,
    automaton: Arc<A>,
    // For JSON fields, the term dictionary include terms from all paths.
    // We apply additional filtering based on the given JSON path, when searching within the term
    // dictionary. This prevents terms from unrelated paths from matching the search criteria.
    json_path_bytes: Option<Box<[u8]>>,
    max_expansions: Option<u32>,
    fuzzy_scoring: bool,
}

impl<A> AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    /// Create a new AutomationWeight
    pub fn new<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        max_expansions: Option<u32>,
        fuzzy_scoring: bool,
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: None,
            max_expansions,
            fuzzy_scoring,
        }
    }

    /// Create a new AutomationWeight for a json path
    pub fn new_for_json_path<IntoArcA: Into<Arc<A>>>(
        field: Field,
        automaton: IntoArcA,
        json_path_bytes: &[u8],
        max_expansions: Option<u32>,
        fuzzy_scoring: bool,
    ) -> AutomatonWeight<A> {
        AutomatonWeight {
            field,
            automaton: automaton.into(),
            json_path_bytes: Some(json_path_bytes.to_vec().into_boxed_slice()),
            max_expansions,
            fuzzy_scoring,
        }
    }

    fn automaton_stream<'a>(
        &'a self,
        term_dict: &'a TermDictionary,
    ) -> io::Result<TermWithStateStreamer<'a, &'a A>> {
        let automaton: &A = &self.automaton;
        let mut term_stream_builder = term_dict.search_with_state(automaton);

        if let Some(json_path_bytes) = &self.json_path_bytes {
            term_stream_builder = term_stream_builder.ge(json_path_bytes);
            if let Some(end) = prefix_end(json_path_bytes) {
                term_stream_builder = term_stream_builder.lt(&end);
            }
        }

        term_stream_builder.into_stream()
    }

    /// Returns the term infos that match the automaton
    pub fn get_match_term_infos(&self, reader: &SegmentReader) -> crate::Result<Vec<TermInfo>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let mut term_infos = Vec::new();
        while term_stream.advance() {
            term_infos.push(term_stream.value().clone());
        }
        Ok(term_infos)
    }

    fn add_term_to_bitset(
        &self,
        term_info: &TermInfo,
        inverted_index: &Arc<crate::InvertedIndexReader>,
        doc_bitset: &mut BitSet,
    ) -> Result<(), TantivyError> {
        let mut block_segment_postings =
            inverted_index.read_block_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;

        loop {
            let docs = block_segment_postings.docs();
            if docs.is_empty() {
                break;
            }
            for &doc in docs {
                doc_bitset.insert(doc);
            }
            block_segment_postings.advance();
        }
        Ok(())
    }

    fn push_term_scorer(
        &self,
        score: Score,
        term_info: &TermInfo,
        inverted_index: &Arc<crate::InvertedIndexReader>,
        boost: Score,
        scorers: &mut Vec<ConstScorer<crate::postings::SegmentPostings>>,
    ) -> Result<(), TantivyError> {
        let segment_postings =
            inverted_index.read_postings_from_terminfo(term_info, IndexRecordOption::Basic)?;
        scorers.push(ConstScorer::new(segment_postings, boost * score));
        Ok(())
    }

    /// Select up to `max_expansions` matching terms, keeping those closest to
    /// the query (highest automaton score, i.e. lowest edit distance) rather
    /// than the first `max_expansions` terms in the dictionary.
    ///
    /// Ties are broken toward the term appearing earlier in the
    /// (lexicographically sorted) term stream, so the selection is
    /// deterministic and independent of how many terms precede the good ones.
    fn select_top_terms(
        &self,
        term_stream: &mut TermWithStateStreamer<'_, &A>,
        max_expansions: u32,
    ) -> Vec<ScoredTermInfo> {
        let capacity = max_expansions as usize;
        if capacity == 0 {
            return Vec::new();
        }
        // A max-heap whose top is the most-evictable term: the lowest score,
        // and among equal scores the one appearing later in the stream.
        //
        // Because terms are streamed in order, each candidate has the largest
        // `order` seen so far, so on an equal score it can never displace an
        // earlier-seen term. A candidate therefore enters a full heap iff its
        // score is strictly greater than the current minimum — which lets us
        // test cheaply and clone the `TermInfo` only for terms we actually keep.
        let mut heap: BinaryHeap<ScoredTermInfo> = BinaryHeap::new();
        let mut order: u64 = 0;
        while term_stream.advance() {
            let score = term_stream
                .state()
                .map(|state| automaton_score(self.automaton.as_ref(), state))
                .unwrap_or(1.0);
            let keep = if heap.len() < capacity {
                true
            } else {
                // `capacity >= 1`, so the heap is non-empty when full.
                heap.peek().is_some_and(|top| score > top.score)
            };
            if keep {
                if heap.len() == capacity {
                    heap.pop();
                }
                heap.push(ScoredTermInfo {
                    score,
                    order,
                    term_info: term_stream.value().clone(),
                });
            }
            order += 1;
        }
        heap.into_vec()
    }
}

/// A matching term paired with its automaton score, used to keep only the
/// closest `max_expansions` terms.
struct ScoredTermInfo {
    score: Score,
    /// Position of the term in the sorted term stream (lower = earlier). Used
    /// as a deterministic tie-breaker between terms of equal score.
    order: u64,
    term_info: TermInfo,
}

impl PartialEq for ScoredTermInfo {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order && self.score == other.score
    }
}

impl Eq for ScoredTermInfo {}

impl PartialOrd for ScoredTermInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredTermInfo {
    /// Orders by "evictability": the greatest element is the one to drop first
    /// when the heap is full — the lowest score, and among equal scores the
    /// term appearing later in the stream (higher `order`).
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.order.cmp(&other.order))
    }
}

impl<A> Weight for AutomatonWeight<A>
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let inverted_index = reader.inverted_index(self.field)?;
        let term_dict = inverted_index.terms();
        let mut term_stream = self.automaton_stream(term_dict)?;
        let max_doc = reader.max_doc();
        if self.fuzzy_scoring {
            let mut scorers = vec![];
            match self.max_expansions {
                // Cap present: keep only the closest terms by edit distance.
                Some(max_expansions) => {
                    for scored in self.select_top_terms(&mut term_stream, max_expansions) {
                        self.push_term_scorer(
                            scored.score,
                            &scored.term_info,
                            &inverted_index,
                            boost,
                            &mut scorers,
                        )?;
                    }
                }
                // No cap: expand every matching term.
                None => {
                    while term_stream.advance() {
                        if let Some(state) = term_stream.state() {
                            let score = automaton_score(self.automaton.as_ref(), state);
                            self.push_term_scorer(
                                score,
                                term_stream.value(),
                                &inverted_index,
                                boost,
                                &mut scorers,
                            )?;
                        }
                    }
                }
            }

            let scorer = super::BufferedUnionScorer::build(scorers, SumCombiner::default, max_doc);
            Ok(Box::new(scorer))
        } else {
            let mut doc_bitset = BitSet::with_max_value(max_doc);
            match self.max_expansions {
                // Cap present: keep only the closest terms by edit distance.
                Some(max_expansions) => {
                    for scored in self.select_top_terms(&mut term_stream, max_expansions) {
                        self.add_term_to_bitset(
                            &scored.term_info,
                            &inverted_index,
                            &mut doc_bitset,
                        )?;
                    }
                }
                // No cap: include every matching term.
                None => {
                    while term_stream.advance() {
                        self.add_term_to_bitset(
                            term_stream.value(),
                            &inverted_index,
                            &mut doc_bitset,
                        )?;
                    }
                }
            }
            let doc_bitset = BitSetDocSet::from(doc_bitset);
            let const_scorer = ConstScorer::new(doc_bitset, boost);
            Ok(Box::new(const_scorer))
        }
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) == doc {
            if self.fuzzy_scoring {
                Ok(Explanation::new("AutomatonScorer", scorer.score()))
            } else {
                Ok(Explanation::new("AutomatonScorer", 1.0))
            }
        } else {
            Err(TantivyError::InvalidArgument(
                "Document does not exist".to_string(),
            ))
        }
    }
}

fn automaton_score<A>(automaton: &A, state: &A::State) -> f32
where
    A: Automaton + Send + Sync + 'static,
    A::State: Clone,
{
    if TypeId::of::<DfaWrapper>() == automaton.type_id() && TypeId::of::<u32>() == state.type_id() {
        let dfa = automaton as *const A as *const DfaWrapper;
        let dfa = unsafe { &*dfa };
        let id = state as *const A::State as *const u32;
        let id = unsafe { *id };
        let dist = dfa.0.distance(id).to_u8() as f32;
        1.0 / (1.0 + dist)
    } else if TypeId::of::<
        super::fuzzy_query::Intersection<
            DfaWrapper,
            StartsWithAutomaton<super::fuzzy_query::Str, Option<usize>>,
            <DfaWrapper as tantivy_fst::Automaton>::State,
            <StartsWithAutomaton<super::fuzzy_query::Str, Option<usize>> as tantivy_fst::Automaton>::State,
        >,
    >() == automaton.type_id()
        && TypeId::of::<IntersectionState<u32, StartsWithAutomatonState<Option<usize>>>>() == state.type_id()
    {
        let dfa = automaton as *const A
            as *const super::fuzzy_query::Intersection<
            DfaWrapper,
            StartsWithAutomaton<super::fuzzy_query::Str, Option<usize>>,
            <DfaWrapper as tantivy_fst::Automaton>::State,
            <StartsWithAutomaton<super::fuzzy_query::Str, Option<usize>> as tantivy_fst::Automaton>::State,
        >;
        let dfa = unsafe { &*dfa };
        let id = state as *const A::State as *const IntersectionState<u32, StartsWithAutomatonState<Option<usize>>>;
        let id = unsafe { &*id };
        let dist = dfa.automaton_a.0.distance(id.0).to_u8() as f32;
        1.0 / (1.0 + dist)
    } else {
        1.0
    }
}
#[cfg(test)]
mod tests {
    use tantivy_fst::Automaton;

    use super::AutomatonWeight;
    use crate::docset::TERMINATED;
    use crate::query::Weight;
    use crate::schema::{Schema, STRING};
    use crate::{Index, IndexWriter};

    fn create_index() -> crate::Result<Index> {
        let mut schema = Schema::builder();
        let title = schema.add_text_field("title", STRING);
        let index = Index::create_in_ram(schema.build());
        let mut index_writer: IndexWriter = index.writer_for_tests()?;
        index_writer.add_document(doc!(title=>"abc"))?;
        index_writer.add_document(doc!(title=>"bcd"))?;
        index_writer.add_document(doc!(title=>"abcd"))?;
        index_writer.commit()?;
        Ok(index)
    }

    #[derive(Clone, Copy)]
    enum State {
        Start,
        NotMatching,
        AfterA,
    }

    struct PrefixedByA;

    impl Automaton for PrefixedByA {
        type State = State;

        fn start(&self) -> Self::State {
            State::Start
        }

        fn is_match(&self, state: &Self::State) -> bool {
            matches!(*state, State::AfterA)
        }

        fn accept(&self, state: &Self::State, byte: u8) -> Self::State {
            match *state {
                State::Start => {
                    if byte == b'a' {
                        State::AfterA
                    } else {
                        State::NotMatching
                    }
                }
                State::AfterA => State::AfterA,
                State::NotMatching => State::NotMatching,
            }
        }
    }

    #[test]
    fn test_automaton_weight() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA, None, false);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.0)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), 2u32);
        assert_eq!(scorer.doc(), 2u32);
        assert_eq!(scorer.score(), 1.0);
        assert_eq!(scorer.advance(), TERMINATED);
        Ok(())
    }

    #[test]
    fn test_automaton_weight_boost() -> crate::Result<()> {
        let index = create_index()?;
        let field = index.schema().get_field("title").unwrap();
        let automaton_weight = AutomatonWeight::new(field, PrefixedByA, None, false);
        let reader = index.reader()?;
        let searcher = reader.searcher();
        let mut scorer = automaton_weight.scorer(searcher.segment_reader(0u32), 1.32)?;
        assert_eq!(scorer.doc(), 0u32);
        assert_eq!(scorer.score(), 1.32);
        Ok(())
    }
}
