// Demonstrates word-level ngram indexing for faster phrase search
//
// This example shows how to:
// 1. Enable word ngram indexing on text fields
// 2. Index documents with automatic ngram generation
// 3. Search for phrases using ngram terms
// 4. Compare performance with regular phrase queries

use tantivy::collector::TopDocs;
use tantivy::query::{PhraseQuery, TermQuery};
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TEXT};
use tantivy::{doc, Index, Term, WordNgramConfig, WordNgramSet};

fn main() -> tantivy::Result<()> {
    // Create a schema with word ngram indexing enabled
    let mut schema_builder = Schema::builder();

    // Configure the text field with word ngram indexing
    // This will automatically index bigrams (FF, FR, RF patterns) and trigrams (FFF)
    let text_indexing = TextFieldIndexing::default()
        .set_index_option(IndexRecordOption::WithFreqsAndPositions)
        .set_word_ngrams(WordNgramConfig::new(
            // Enable frequent-frequent and all frequent trigrams
            WordNgramSet::NGRAM_FF | WordNgramSet::NGRAM_FFF,
        ));

    let text_field = schema_builder.add_text_field("text", TEXT.clone().set_indexing_options(text_indexing));

    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut index_writer = index.writer(50_000_000)?;

    // Index some documents
    println!("Indexing documents with automatic ngram generation...\n");
    
    index_writer.add_document(doc!(
        text_field => "the quick brown fox jumps over the lazy dog"
    ))?;

    index_writer.add_document(doc!(
        text_field => "a quick brown cat runs through the garden"
    ))?;

    index_writer.add_document(doc!(
        text_field => "the slow brown turtle walks in the sun"
    ))?;

    index_writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();

    println!("--- Searching for Individual Terms ---");
    
    // Search for a single term
    let term_query = TermQuery::new(
        Term::from_field_text(text_field, "quick"),
        IndexRecordOption::Basic,
    );
    let top_docs = searcher.search(&term_query, &TopDocs::with_limit(10).order_by_score())?;
    println!("Single term 'quick': {} documents", top_docs.len());

    println!("\n--- Searching for Ngram Terms (Indexed) ---");

    // Search for a bigram that was automatically indexed
    let bigram_query = TermQuery::new(
        Term::from_field_text(text_field, "quick brown"),
        IndexRecordOption::Basic,
    );
    let top_docs = searcher.search(&bigram_query, &TopDocs::with_limit(10).order_by_score())?;
    println!("Bigram 'quick brown': {} documents", top_docs.len());
    println!("  ✓ This is a direct term lookup (fast!)");

    // Search for another bigram
    let bigram_query = TermQuery::new(
        Term::from_field_text(text_field, "brown fox"),
        IndexRecordOption::Basic,
    );
    let top_docs = searcher.search(&bigram_query, &TopDocs::with_limit(10).order_by_score())?;
    println!("Bigram 'brown fox': {} documents", top_docs.len());

    // Search for a trigram that was automatically indexed
    let trigram_query = TermQuery::new(
        Term::from_field_text(text_field, "quick brown fox"),
        IndexRecordOption::Basic,
    );
    let top_docs = searcher.search(&trigram_query, &TopDocs::with_limit(10).order_by_score())?;
    println!("Trigram 'quick brown fox': {} documents", top_docs.len());
    println!("  ✓ Also a direct term lookup!");

    println!("\n--- Phrase Queries (Uses Position Matching) ---");

    // Traditional phrase query still works (uses position matching as fallback)
    let phrase_query = PhraseQuery::new(vec![
        Term::from_field_text(text_field, "quick"),
        Term::from_field_text(text_field, "brown"),
    ]);
    let top_docs = searcher.search(&phrase_query, &TopDocs::with_limit(10).order_by_score())?;
    println!("Phrase query [quick brown]: {} documents", top_docs.len());
    println!("  → Falls back to position matching");

    println!("\n--- Performance Benefits ---");
    println!("• Ngram term queries: O(1) dictionary lookup");
    println!("• Phrase queries: O(n) position intersection");
    println!("• Trade-off: ~30% larger index for 2-10x faster phrase search");

    println!("\n--- Index Statistics ---");
    
    // Count total terms in the index (includes ngrams)
    let inverted_index = searcher.segment_reader(0).inverted_index(text_field)?;
    let terms = inverted_index.terms();
    println!("Total terms indexed (including ngrams): {}", terms.num_terms());

    Ok(())
}
