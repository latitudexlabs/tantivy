// Example: Using Word Ngram Indexing for Fast Phrase Search
//
// This example demonstrates how to enable and use frequency-based word ngram indexing
// to dramatically speed up phrase queries.

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{STORED, Schema, TEXT, TextFieldIndexing, Value};
use tantivy::tokenizer::WhitespaceTokenizer;
use tantivy::{doc, Index, IndexWriter, TantivyDocument, WordNgramConfig, WordNgramSet};

fn main() -> tantivy::Result<()> {
    // Create a schema with word ngram indexing enabled
    let mut schema_builder = Schema::builder();

    // Configure the text field with word ngram indexing
    // This will create bigrams for various frequency patterns and trigrams for frequent triplets
    // For minimal phrase query latency at the cost of larger index size
    let text_indexing = TextFieldIndexing::default()
        .set_tokenizer("whitespace")
        .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions)
        .set_word_ngrams(
            WordNgramConfig::with_set(
                // Enable mixed bigrams (FF, FR, RF) and frequent trigrams (FFF)
                // This provides the best phrase query performance
                WordNgramSet::new()
                    .with_ngram_ff()   // Frequent-Frequent bigrams
                    .with_ngram_fr()   // Frequent-Rare bigrams
                    .with_ngram_rf()   // Rare-Frequent bigrams
                    .with_ngram_fff(), // Frequent-Frequent-Frequent trigrams
            )
            // Set the threshold: terms appearing in >1% of docs are "frequent"
            .with_frequent_threshold(0.01)
            // Track up to 10,000 frequent terms
            .with_max_frequent_terms(10_000),
        );

    let body = schema_builder.add_text_field(
        "body",
        TEXT.clone().set_indexing_options(text_indexing) | STORED,
    );

    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema.clone());

    // Register the whitespace tokenizer
    index
        .tokenizers()
        .register("whitespace", WhitespaceTokenizer::default());

    // Create an index writer
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;

    // Index some documents
    let documents = vec![
        "the quick brown fox jumps over the lazy dog",
        "the dog was very lazy in the afternoon sun",
        "a quick fox is faster than a lazy dog",
        "in the world of animals, the fox is quick",
        "the lazy cat sleeps in the sun all day",
        "the world is full of quick and lazy animals",
        "the dog and the cat are best friends in the world",
        "the quick brown rabbit hops in the garden",
        "in the garden, the lazy dog rests under a tree",
        "the world needs more quick thinking and less lazy behavior",
        "no no",
        "no no no",
    ];

    println!("Indexing {} documents...", documents.len());
    for text in documents {
        index_writer.add_document(doc!(body => text))?;
    }

    // Commit to make documents searchable
    index_writer.commit()?;

    println!("\n✓ Indexing complete!\n");

    // Search the index
    let reader = index.reader()?;
    let searcher = reader.searcher();

    // Create a query parser
    let query_parser = QueryParser::for_index(&index, vec![body]);

    // Example queries that benefit from ngram indexing
    let queries = vec![
        "\"the lazy\"",     // Phrase query - bigram
        "\"in the world\"", // Phrase query - trigram
        "\"the quick\"",    // Phrase query - bigram
        "quick lazy",       // Term query (for comparison)
        "\"no no no\""      // Phrase query - trigram
    ];

    for query_str in queries {
        println!("Query: {}", query_str);
        let query = query_parser.parse_query(query_str)?;
        let top_docs: Vec<(f32, tantivy::DocAddress)> = 
            searcher.search(&query, &TopDocs::with_limit(5).order_by_score())?;

        println!("  Found {} results:", top_docs.len());
        for (_score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(body_value) = retrieved_doc.get_first(body) {
                if let Some(text) = body_value.as_str() {
                    println!("    - {}", text);
                }
            }
        }
        println!();
    }
    Ok(())
}
