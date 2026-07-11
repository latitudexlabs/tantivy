use std::collections::{HashSet, VecDeque};
use std::io;
use std::marker::PhantomData;
use std::ops::Range;

use stacker::Addr;

use crate::fieldnorm::FieldNormReaders;
use crate::indexer::doc_id_mapping::DocIdMapping;
use crate::indexer::indexing_term::IndexingTerm;
use crate::indexer::path_to_unordered_id::OrderedPathId;
use crate::indexer::WordNgramConfig;
use crate::postings::recorder::{BufferLender, Recorder};
use crate::postings::{
    FieldSerializer, IndexingContext, InvertedIndexSerializer, PerFieldPostingsWriter,
};
use crate::schema::{Field, Schema, Type};
use crate::tokenizer::{Token, TokenStream, MAX_TOKEN_LEN};
use crate::DocId;

const POSITION_GAP: u32 = 1;

/// Number of token positions kept in the sliding window when generating
/// consecutive (non-`all_combinations`) ngrams: at most the last three tokens
/// are needed to emit adjacent bigrams and trigrams.
const CONSECUTIVE_NGRAM_WINDOW: usize = 3;

fn make_field_partition(
    term_offsets: &[(Field, OrderedPathId, &[u8], Addr)],
) -> Vec<(Field, Range<usize>)> {
    let term_offsets_it = term_offsets
        .iter()
        .map(|(field, _, _, _)| *field)
        .enumerate();
    let mut prev_field_opt = None;
    let mut fields = vec![];
    let mut offsets = vec![];
    for (offset, field) in term_offsets_it {
        if Some(field) != prev_field_opt {
            prev_field_opt = Some(field);
            fields.push(field);
            offsets.push(offset);
        }
    }
    offsets.push(term_offsets.len());
    let mut field_offsets = vec![];
    for i in 0..fields.len() {
        field_offsets.push((fields[i], offsets[i]..offsets[i + 1]));
    }
    field_offsets
}

/// Serialize the inverted index.
/// It pushes all term, one field at a time, towards the
/// postings serializer.
pub(crate) fn serialize_postings(
    ctx: IndexingContext,
    schema: Schema,
    per_field_postings_writers: &PerFieldPostingsWriter,
    fieldnorm_readers: FieldNormReaders,
    doc_id_map: Option<&DocIdMapping>,
    serializer: &mut InvertedIndexSerializer,
) -> crate::Result<()> {
    // Replace unordered ids by ordered ids to be able to sort
    let unordered_id_to_ordered_id: Vec<OrderedPathId> =
        ctx.path_to_unordered_id.unordered_id_to_ordered_id();

    let mut term_offsets: Vec<(Field, OrderedPathId, &[u8], Addr)> =
        Vec::with_capacity(ctx.term_index.len());
    term_offsets.extend(ctx.term_index.iter().map(|(key, addr)| {
        let field = IndexingTerm::wrap(key).field();
        if schema.get_field_entry(field).field_type().value_type() == Type::Json {
            let byte_range_path = 4..4 + 4;
            let unordered_id = u32::from_be_bytes(key[byte_range_path.clone()].try_into().unwrap());
            let path_id = unordered_id_to_ordered_id[unordered_id as usize];
            (field, path_id, &key[byte_range_path.end..], addr)
        } else {
            (field, 0.into(), &key[4..], addr)
        }
    }));
    // Sort by field, path, and term
    term_offsets.sort_unstable_by(
        |(field1, path_id1, bytes1, _), (field2, path_id2, bytes2, _)| {
            (field1, path_id1, bytes1).cmp(&(field2, path_id2, bytes2))
        },
    );
    let ordered_id_to_path = ctx.path_to_unordered_id.ordered_id_to_path();
    let field_offsets = make_field_partition(&term_offsets);
    for (field, byte_offsets) in field_offsets {
        let postings_writer = per_field_postings_writers.get_for_field(field);
        let fieldnorm_reader = fieldnorm_readers.get_field(field)?;
        let mut field_serializer =
            serializer.new_field(field, postings_writer.total_num_tokens(), fieldnorm_reader)?;
        postings_writer.serialize(
            &term_offsets[byte_offsets],
            &ordered_id_to_path,
            doc_id_map,
            &ctx,
            &mut field_serializer,
        )?;
        field_serializer.close()?;
    }

    Ok(())
}

#[derive(Default, Debug)]
pub(crate) struct IndexingPosition {
    pub num_tokens: u32,
    pub end_position: u32,
}

/// The `PostingsWriter` is in charge of receiving documenting
/// and building a `Segment` in anonymous memory.
///
/// `PostingsWriter` writes in a `MemoryArena`.
pub(crate) trait PostingsWriter: Send + Sync {
    /// Record that a document contains a term at a given position.
    ///
    /// * doc  - the document id
    /// * pos  - the term position (expressed in tokens)
    /// * term - the term
    /// * ctx - Contains a term hashmap and a memory arena to store all necessary posting list
    ///   information.
    fn subscribe(&mut self, doc: DocId, pos: u32, term: &IndexingTerm, ctx: &mut IndexingContext);

    /// Serializes the postings on disk.
    /// The actual serialization format is handled by the `PostingsSerializer`.
    fn serialize(
        &self,
        term_addrs: &[(Field, OrderedPathId, &[u8], Addr)],
        ordered_id_to_path: &[&str],
        doc_id_map: Option<&DocIdMapping>,
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()>;

    /// Tokenize a text and subscribe all of its token.
    fn index_text(
        &mut self,
        doc_id: DocId,
        token_stream: &mut dyn TokenStream,
        term_buffer: &mut IndexingTerm,
        ctx: &mut IndexingContext,
        indexing_position: &mut IndexingPosition,
    ) {
        let end_of_path_idx = term_buffer.len_bytes();
        let mut num_tokens = 0;
        let mut end_position = indexing_position.end_position;
        token_stream.process(&mut |token: &Token| {
            // We skip all tokens with a len greater than u16.
            if token.text.len() > MAX_TOKEN_LEN {
                warn!(
                    "A token exceeding MAX_TOKEN_LEN ({}>{}) was dropped. Search for \
                     MAX_TOKEN_LEN in the documentation for more information.",
                    token.text.len(),
                    MAX_TOKEN_LEN
                );
                return;
            }
            term_buffer.truncate_value_bytes(end_of_path_idx);
            term_buffer.append_bytes(token.text.as_bytes());
            let start_position = indexing_position.end_position + token.position as u32;
            end_position = end_position.max(start_position + token.position_length as u32);
            self.subscribe(doc_id, start_position, term_buffer, ctx);
            num_tokens += 1;
        });

        indexing_position.end_position = end_position + POSITION_GAP;
        indexing_position.num_tokens += num_tokens;
        term_buffer.truncate_value_bytes(end_of_path_idx);
    }

    fn total_num_tokens(&self) -> u64;
}

/// The `SpecializedPostingsWriter` is just here to remove dynamic
/// dispatch to the recorder information.
pub(crate) struct SpecializedPostingsWriter<Rec: Recorder> {
    total_num_tokens: u64,
    _recorder_type: PhantomData<Rec>,
    /// Sliding window of recent term texts for ngram generation
    /// When edge_ngram is enabled, each element is a Vec of edge ngram variants for that token position
    term_window: VecDeque<Vec<String>>,
    /// Word ngram configuration if enabled
    ngram_config: Option<WordNgramConfig>,
    /// Reusable buffer for ngram construction (avoids repeated allocations)
    ngram_buffer: String,
    /// Set to track which ngrams have been emitted for the current field to avoid duplicates
    seen_ngrams: HashSet<String>,
}

impl<Rec: Recorder> Default for SpecializedPostingsWriter<Rec> {
    fn default() -> Self {
        Self {
            total_num_tokens: 0,
            _recorder_type: PhantomData,
            term_window: VecDeque::with_capacity(3),
            ngram_config: None,
            ngram_buffer: String::with_capacity(64),
            seen_ngrams: HashSet::new(),
        }
    }
}

impl<Rec: Recorder> From<SpecializedPostingsWriter<Rec>> for Box<dyn PostingsWriter> {
    fn from(
        specialized_postings_writer: SpecializedPostingsWriter<Rec>,
    ) -> Box<dyn PostingsWriter> {
        Box::new(specialized_postings_writer)
    }
}

impl<Rec: Recorder> SpecializedPostingsWriter<Rec> {
    /// Create a new postings writer with optional ngram configuration
    pub(crate) fn with_ngram_config(ngram_config: Option<WordNgramConfig>) -> Self {
        // Pre-size the sliding window to the maximum number of token groups it
        // will hold while indexing, so it does not reallocate as tokens stream
        // in. `index_text` pushes one group before trimming, so the transient
        // peak is one above the steady-state window size.
        let window_capacity = match &ngram_config {
            Some(config) if config.contains_bigrams() || config.contains_trigrams() => {
                let window = if config.all_combinations {
                    config.all_combinations_window_size
                } else {
                    CONSECUTIVE_NGRAM_WINDOW
                };
                window.saturating_add(1)
            }
            // No ngrams: the window is never populated.
            _ => 0,
        };
        Self {
            total_num_tokens: 0,
            _recorder_type: PhantomData,
            term_window: VecDeque::with_capacity(window_capacity),
            ngram_config,
            ngram_buffer: String::with_capacity(64),
            seen_ngrams: HashSet::new(),
        }
    }

    #[inline]
    pub(crate) fn serialize_one_term(
        term: &[u8],
        addr: Addr,
        doc_id_map: Option<&DocIdMapping>,
        buffer_lender: &mut BufferLender,
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()> {
        let recorder: Rec = ctx.term_index.read(addr);
        let term_doc_freq = recorder.term_doc_freq().unwrap_or(0u32);
        serializer.new_term(term, term_doc_freq, recorder.has_term_freq())?;
        recorder.serialize(&ctx.arena, doc_id_map, serializer, buffer_lender);
        serializer.close_term()?;
        Ok(())
    }
}

impl<Rec: Recorder> PostingsWriter for SpecializedPostingsWriter<Rec> {
    #[inline]
    fn subscribe(
        &mut self,
        doc: DocId,
        position: u32,
        term: &IndexingTerm,
        ctx: &mut IndexingContext,
    ) {
        debug_assert!(term.serialized_term().len() >= 4);
        self.total_num_tokens += 1;
        let (term_index, arena) = (&mut ctx.term_index, &mut ctx.arena);
        term_index.mutate_or_create(term.serialized_term(), |opt_recorder: Option<Rec>| {
            if let Some(mut recorder) = opt_recorder {
                let current_doc = recorder.current_doc();
                if current_doc != doc {
                    recorder.close_doc(arena);
                    recorder.new_doc(doc, arena);
                }
                recorder.record_position(position, arena);
                recorder
            } else {
                let mut recorder = Rec::default();
                recorder.new_doc(doc, arena);
                recorder.record_position(position, arena);
                recorder
            }
        });
    }

    fn index_text(
        &mut self,
        doc_id: DocId,
        token_stream: &mut dyn TokenStream,
        term_buffer: &mut IndexingTerm,
        ctx: &mut IndexingContext,
        indexing_position: &mut IndexingPosition,
    ) {
        let end_of_path_idx = term_buffer.len_bytes();
        let mut num_tokens = 0;
        let mut end_position = indexing_position.end_position;

        // Clear term window and seen ngrams at the start of each field value
        self.term_window.clear();
        self.seen_ngrams.clear();

        // Extract config flags once outside the loop for better performance
        let (
            should_gen_bigrams,
            should_gen_trigrams,
            all_combinations,
            combinations_window_size,
            edge_ngram_enabled,
            min_edge_ngram,
        ) = if let Some(ref config) = self.ngram_config {
            (
                config.contains_bigrams(),
                config.contains_trigrams(),
                config.all_combinations,
                config.all_combinations_window_size,
                config.edge_ngram,
                config.min_edge_ngram,
            )
        } else {
            (false, false, false, 5, false, 2)
        };

        token_stream.process(&mut |token: &Token| {
            // We skip all tokens with a len greater than u16.
            if token.text.len() > MAX_TOKEN_LEN {
                warn!(
                    "A token exceeding MAX_TOKEN_LEN ({}>{}) was dropped. Search for \
                     MAX_TOKEN_LEN in the documentation for more information.",
                    token.text.len(),
                    MAX_TOKEN_LEN
                );
                return;
            }

            // Index the original term
            term_buffer.truncate_value_bytes(end_of_path_idx);
            term_buffer.append_bytes(token.text.as_bytes());
            let start_position = indexing_position.end_position + token.position as u32;
            end_position = end_position.max(start_position + token.position_length as u32);
            self.subscribe(doc_id, start_position, term_buffer, ctx);
            num_tokens += 1;

            // Generate ngrams if configured
            // Note: We generate ALL ngrams for configured types (FF, FR, RF, etc.) without
            // filtering by actual frequency during indexing. Frequency-based selection happens
            // at query time when the FrequentTermTracker data is available.
            if should_gen_bigrams || should_gen_trigrams {
                // Build this token's edge-ngram group. It is moved into the
                // sliding window below, so a fresh allocation is unavoidable
                // per token; size it exactly to avoid intermediate reallocations.
                let edge_ngrams: Vec<String> = if edge_ngram_enabled {
                    // Progressive prefixes from min_edge_ngram up to the full token.
                    let token_text = &token.text;
                    let char_count = token_text.chars().count();
                    let mut edges =
                        Vec::with_capacity(1 + char_count.saturating_sub(min_edge_ngram));

                    // Always include the full token.
                    edges.push(token_text.to_string());

                    // Generate prefixes from full_len-1 down to min_edge_ngram.
                    if char_count > min_edge_ngram {
                        // Build a char index map for efficient prefix extraction.
                        let char_indices: Vec<usize> = token_text
                            .char_indices()
                            .map(|(i, _)| i)
                            .chain(std::iter::once(token_text.len()))
                            .collect();

                        for prefix_len in (min_edge_ngram..char_count).rev() {
                            edges.push(token_text[..char_indices[prefix_len]].to_string());
                        }
                    }
                    edges
                } else {
                    // No edge ngrams, just use the full token.
                    vec![token.text.to_string()]
                };

                // Add edge ngrams as a group to the sliding window (move instead of clone)
                // This ensures we don't generate combinations within the same token position
                self.term_window.push_back(edge_ngrams);

                // Keep window size limited based on mode
                let max_window_size = if all_combinations {
                    // Use configured window size for all_combinations mode
                    combinations_window_size
                } else {
                    // In consecutive mode, only need the last few tokens
                    CONSECUTIVE_NGRAM_WINDOW
                };

                while self.term_window.len() > max_window_size {
                    self.term_window.pop_front();
                }

                let window_len = self.term_window.len();

                if all_combinations {
                    // Generate ordered combinations that include the newest token position
                    // This is much more efficient than regenerating all combinations
                    // For a window of size N, we generate:
                    // - Bigrams: N-1 pairs (new token position with each previous position)
                    // - Trigrams: C(N-1,2) triplets (new token with each pair of previous positions)

                    if window_len < 2 {
                        // Nothing to do with less than 2 token positions
                        return;
                    }

                    let last_idx = window_len - 1;
                    // Clone only the last position's edge ngrams once
                    let edges_last = self.term_window[last_idx].clone();

                    // Generate bigrams: pair the new token position with each previous position
                    // For each pair of positions, generate all combinations of their edge ngrams
                    if should_gen_bigrams {
                        for i in 0..last_idx {
                            // Clone edge ngrams for position i once per outer loop iteration
                            let edges_i = self.term_window[i].clone();

                            // Generate all combinations of edge ngrams from position i and last position
                            for edge_i in &edges_i {
                                for edge_last in &edges_last {
                                    // Skip bigrams where both words are identical
                                    if edge_i == edge_last {
                                        continue;
                                    }

                                    self.ngram_buffer.clear();
                                    self.ngram_buffer.push_str(edge_i);
                                    self.ngram_buffer.push(' ');
                                    self.ngram_buffer.push_str(edge_last);

                                    // Only index if we haven't seen this ngram for this field
                                    if self.seen_ngrams.insert(self.ngram_buffer.clone()) {
                                        term_buffer.truncate_value_bytes(end_of_path_idx);
                                        term_buffer.append_bytes(self.ngram_buffer.as_bytes());
                                        self.subscribe(doc_id, 0, term_buffer, ctx);
                                    }
                                }
                            }
                        }
                    }

                    // Generate trigrams: combine the new token position with each pair of previous positions
                    if should_gen_trigrams && window_len >= 3 {
                        for i in 0..(last_idx - 1) {
                            // Clone edge ngrams for position i once per outer loop
                            let edges_i = self.term_window[i].clone();
                            
                            for j in (i + 1)..last_idx {
                                // Clone edge ngrams for position j once per middle loop
                                let edges_j = self.term_window[j].clone();

                                // Generate all combinations of edge ngrams from positions i, j, and last
                                for edge_i in &edges_i {
                                    for edge_j in &edges_j {
                                        for edge_last in &edges_last {
                                            // Skip trigrams where all three words are identical
                                            if edge_i == edge_j && edge_j == edge_last {
                                                continue;
                                            }

                                            self.ngram_buffer.clear();
                                            self.ngram_buffer.push_str(edge_i);
                                            self.ngram_buffer.push(' ');
                                            self.ngram_buffer.push_str(edge_j);
                                            self.ngram_buffer.push(' ');
                                            self.ngram_buffer.push_str(edge_last);

                                            // Only index if we haven't seen this ngram for this field
                                            if self.seen_ngrams.insert(self.ngram_buffer.clone()) {
                                                term_buffer.truncate_value_bytes(end_of_path_idx);
                                                term_buffer.append_bytes(self.ngram_buffer.as_bytes());
                                                self.subscribe(doc_id, 0, term_buffer, ctx);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Generate only consecutive ngrams (original behavior)

                    // Generate consecutive bigrams
                    if should_gen_bigrams && window_len >= 2 {
                        // Clone edge ngrams from last two positions once
                        let edges_prev = self.term_window[window_len - 2].clone();
                        let edges_curr = self.term_window[window_len - 1].clone();

                        // Generate all combinations of edge ngrams from last two positions
                        for edge_prev in &edges_prev {
                            for edge_curr in &edges_curr {
                                // Skip bigrams where both words are identical
                                if edge_prev == edge_curr {
                                    continue;
                                }

                                self.ngram_buffer.clear();
                                self.ngram_buffer.push_str(edge_prev);
                                self.ngram_buffer.push(' ');
                                self.ngram_buffer.push_str(edge_curr);

                                // Only index if we haven't seen this ngram for this field
                                if self.seen_ngrams.insert(self.ngram_buffer.clone()) {
                                    term_buffer.truncate_value_bytes(end_of_path_idx);
                                    term_buffer.append_bytes(self.ngram_buffer.as_bytes());
                                    // Ngrams use position 0 (they represent a phrase, not a single word position)
                                    self.subscribe(doc_id, 0, term_buffer, ctx);
                                }
                            }
                        }
                    }

                    // Generate consecutive trigrams (need at least 3 terms) - only if configured
                    if should_gen_trigrams && window_len >= 3 {
                        // Clone edge ngrams from last three positions once
                        let edges_1 = self.term_window[window_len - 3].clone();
                        let edges_2 = self.term_window[window_len - 2].clone();
                        let edges_3 = self.term_window[window_len - 1].clone();

                        // Generate all combinations of edge ngrams from last three positions
                        for edge_1 in &edges_1 {
                            for edge_2 in &edges_2 {
                                for edge_3 in &edges_3 {
                                    // Skip trigrams where all three words are identical
                                    if edge_1 == edge_2 && edge_2 == edge_3 {
                                        continue;
                                    }

                                    self.ngram_buffer.clear();
                                    self.ngram_buffer.push_str(edge_1);
                                    self.ngram_buffer.push(' ');
                                    self.ngram_buffer.push_str(edge_2);
                                    self.ngram_buffer.push(' ');
                                    self.ngram_buffer.push_str(edge_3);

                                    // Only index if we haven't seen this ngram for this field
                                    if self.seen_ngrams.insert(self.ngram_buffer.clone()) {
                                        term_buffer.truncate_value_bytes(end_of_path_idx);
                                        term_buffer.append_bytes(self.ngram_buffer.as_bytes());
                                        // Ngrams use position 0
                                        self.subscribe(doc_id, 0, term_buffer, ctx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        indexing_position.end_position = end_position + POSITION_GAP;
        indexing_position.num_tokens += num_tokens;
        term_buffer.truncate_value_bytes(end_of_path_idx);
    }

    fn serialize(
        &self,
        term_addrs: &[(Field, OrderedPathId, &[u8], Addr)],
        _ordered_id_to_path: &[&str],
        doc_id_map: Option<&DocIdMapping>,
        ctx: &IndexingContext,
        serializer: &mut FieldSerializer,
    ) -> io::Result<()> {
        let mut buffer_lender = BufferLender::default();
        for (_field, _path_id, term, addr) in term_addrs {
            Self::serialize_one_term(term, *addr, doc_id_map, &mut buffer_lender, ctx, serializer)?;
        }
        Ok(())
    }

    fn total_num_tokens(&self) -> u64 {
        self.total_num_tokens
    }
}
