use crate::docset::DocSet;
use crate::fieldnorm::FieldNormReader;
use crate::postings::Postings;
use crate::query::bm25::Bm25Weight;
use crate::query::Scorer;
use crate::{DocId, Score, TERMINATED};

/// Scorer matching documents where at least `min_match` of the phrase terms
/// appear in query order (gaps allowed).
///
/// Candidate documents are found by driving a min-match union directly over
/// the term postings: a candidate must have at least `min_match` postings on
/// the same document, so the search can skip to the `min_match`-th smallest
/// posting cursor instead of visiting every document of the union.
/// Candidates are then verified with a longest-ordered-chain check over the
/// term positions.
pub struct FuzzyPhraseScorer<TPostings: Postings> {
    // Postings for each phrase term, in query order.
    phrase_postings: Vec<TPostings>,
    min_match: usize,
    match_count: usize,
    fieldnorm_reader: FieldNormReader,
    similarity_weight_opt: Option<Bm25Weight>,
    current_doc: DocId,
    // Reusable position buffers to avoid allocations, one per term.
    position_buffers: Vec<Vec<u32>>,
    // Reusable buffer for the longest-chain computation:
    // chain_tails[d] = smallest position at which a chain of length d + 1
    // can end, over the terms processed so far.
    chain_tails: Vec<u32>,
    // Reusable scratch for pivot selection over the postings cursors.
    docs_scratch: Vec<DocId>,
    size_hint: u32,
    cost: u64,
}

impl<TPostings: Postings> FuzzyPhraseScorer<TPostings> {
    pub fn new(
        phrase_postings: Vec<TPostings>,
        min_match: usize,
        similarity_weight_opt: Option<Bm25Weight>,
        fieldnorm_reader: FieldNormReader,
    ) -> FuzzyPhraseScorer<TPostings> {
        assert!(phrase_postings.len() >= 2);
        assert!(min_match > 0 && min_match <= phrase_postings.len());

        let num_terms = phrase_postings.len();
        // Every matching document consumes at least `min_match` posting
        // entries, so the summed doc frequencies bound the candidate count.
        let postings_len_sum: u64 = phrase_postings
            .iter()
            .map(|postings| postings.size_hint() as u64)
            .sum();
        let candidates_upper_bound = postings_len_sum / min_match as u64;
        // The in-order check rejects a share of the candidates; halving the
        // upper bound mirrors the derating done by other position scorers.
        // Round up so a docset with candidates never hints 0.
        let size_hint = (candidates_upper_bound.div_ceil(2)).min(u32::MAX as u64) as u32;
        // Consuming the docset traverses every posting once and loads
        // positions for each candidate.
        let cost = postings_len_sum + candidates_upper_bound * num_terms as u64;

        let mut scorer = FuzzyPhraseScorer {
            phrase_postings,
            min_match,
            match_count: 0,
            fieldnorm_reader,
            similarity_weight_opt,
            current_doc: TERMINATED,
            position_buffers: vec![Vec::new(); num_terms],
            chain_tails: Vec::new(),
            docs_scratch: vec![0; num_terms],
            size_hint,
            cost,
        };
        scorer.position_from(0);
        scorer
    }

    /// Positions the scorer on the first matching document with id >= `lower`
    /// and returns it, or TERMINATED if there is none.
    fn position_from(&mut self, mut lower: DocId) -> DocId {
        loop {
            let candidate = self.next_candidate(lower);
            if candidate == TERMINATED {
                self.current_doc = TERMINATED;
                return TERMINATED;
            }
            if self.check_candidate(candidate) {
                self.current_doc = candidate;
                return candidate;
            }
            lower = candidate + 1;
        }
    }

    /// Returns the smallest doc >= `lower` on which at least `min_match`
    /// postings coincide, seeking the postings forward as needed.
    fn next_candidate(&mut self, lower: DocId) -> DocId {
        loop {
            self.docs_scratch.clear();
            self.docs_scratch
                .extend(self.phrase_postings.iter().map(|postings| postings.doc()));
            // A candidate needs `min_match` postings on the same doc, so no
            // candidate can be smaller than the `min_match`-th smallest
            // cursor: use it as the pivot.
            let (_, &mut pivot, _) = self.docs_scratch.select_nth_unstable(self.min_match - 1);
            let pivot = pivot.max(lower);
            if pivot == TERMINATED {
                return TERMINATED;
            }
            let mut count_at_pivot = 0;
            for postings in &mut self.phrase_postings {
                if postings.doc() < pivot {
                    postings.seek(pivot);
                }
                if postings.doc() == pivot {
                    count_at_pivot += 1;
                }
            }
            if count_at_pivot >= self.min_match {
                return pivot;
            }
            // Fewer than `min_match` postings on the pivot and none left
            // behind it: the next pivot is strictly larger, so this loop
            // terminates.
        }
    }

    /// Runs the in-order check on `doc` (postings must already be positioned)
    /// and records the resulting `match_count`.
    fn check_candidate(&mut self, doc: DocId) -> bool {
        let scoring_enabled = self.similarity_weight_opt.is_some();
        if self.min_match == 1 && !scoring_enabled {
            // Any present term forms a chain of length 1: no need to decode
            // positions when the count is not used for scoring.
            self.match_count = 1;
            return true;
        }
        for (idx, postings) in self.phrase_postings.iter_mut().enumerate() {
            self.position_buffers[idx].clear();
            if postings.doc() == doc {
                postings.positions(&mut self.position_buffers[idx]);
            }
        }
        // With scoring enabled, the full chain length is the term frequency
        // fed to BM25, so the early exit must not cap it.
        self.match_count = self.longest_ordered_chain(!scoring_enabled);
        self.match_count >= self.min_match
    }

    /// Length of the longest chain of terms appearing in query order with
    /// strictly increasing positions.
    ///
    /// This is a longest-increasing-subsequence over the per-term position
    /// lists: `chain_tails` stays sorted, and `chain_tails[d]` is the
    /// smallest position ending a chain of length `d + 1`. Each term's
    /// positions are processed in descending order so that a term cannot
    /// chain with itself: an update from a larger position can only raise
    /// entries above any smaller position of the same term, never changing
    /// the depth computed for it.
    fn longest_ordered_chain(&mut self, early_exit: bool) -> usize {
        self.chain_tails.clear();
        for positions in &self.position_buffers {
            for &position in positions.iter().rev() {
                let depth = self.chain_tails.partition_point(|&tail| tail < position);
                if depth == self.chain_tails.len() {
                    self.chain_tails.push(position);
                    if early_exit && self.chain_tails.len() >= self.min_match {
                        return self.min_match;
                    }
                } else if position < self.chain_tails[depth] {
                    self.chain_tails[depth] = position;
                }
            }
        }
        self.chain_tails.len()
    }

    #[inline]
    pub(crate) fn match_count(&self) -> u32 {
        self.match_count as u32
    }
}

impl<TPostings: Postings> DocSet for FuzzyPhraseScorer<TPostings> {
    fn advance(&mut self) -> DocId {
        if self.current_doc == TERMINATED {
            return TERMINATED;
        }
        self.position_from(self.current_doc + 1)
    }

    fn seek(&mut self, target: DocId) -> DocId {
        // Covers both target == current_doc (already positioned) and a
        // terminated docset (current_doc == TERMINATED >= any target).
        if target <= self.current_doc {
            return self.current_doc;
        }
        self.position_from(target)
    }

    fn doc(&self) -> DocId {
        self.current_doc
    }

    fn size_hint(&self) -> u32 {
        self.size_hint
    }

    fn cost(&self) -> u64 {
        self.cost
    }
}

impl<TPostings: Postings> Scorer for FuzzyPhraseScorer<TPostings> {
    fn score(&mut self) -> Score {
        if let Some(similarity_weight) = self.similarity_weight_opt.as_ref() {
            let fieldnorm_id = self.fieldnorm_reader.fieldnorm_id(self.doc());
            similarity_weight.score(fieldnorm_id, self.match_count as u32)
        } else {
            1.0
        }
    }
}
