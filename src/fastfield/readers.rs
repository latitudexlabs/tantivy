use std::collections::HashMap;
use std::io;
use std::net::Ipv6Addr;
use std::sync::{Arc, RwLock};

use columnar::{
    BytesColumn, Column, ColumnType, ColumnValues, ColumnarReader, DynamicColumn,
    DynamicColumnHandle, HasAssociatedColumnType, StrColumn,
};
use common::ByteCount;

use crate::core::json_utils::{encode_column_name, json_path_sep_to_dot};
use crate::directory::FileSlice;
use crate::schema::{Field, FieldEntry, FieldType, Schema};
use crate::space_usage::{FieldUsage, PerFieldSpaceUsage};
use crate::TantivyError;

/// Memoizes the per-segment fast field lookups that are pure functions of the
/// (immutable) columnar file and schema.
///
/// Resolving a fast field goes through three steps, and *every* one of them was
/// repeated on every query touching that field:
///  1. `resolve_field` + `ColumnarReader::read_columns` — an sstable lookup
///     yielding the column handles for a field name.
///  2. `DynamicColumnHandle::open` — parses the column header and materializes
///     the column index. For a blockwise-linear index this rebuilds the whole
///     per-block metadata vector, which dominated the cost.
///  3. the same, through the `u64`-lenient view used by range queries.
///
/// A segment's columnar data never changes once opened, so the results are
/// stable and safe to keep. Cached values (`DynamicColumnHandle`, `DynamicColumn`,
/// `Column<u64>`) are all cheap `Arc`/`OwnedBytes` clones, so a hit costs a lock
/// plus a refcount bump instead of a re-parse.
///
/// "This field has no column of that type" is cached as `None` — that is bounded
/// by the segment's own fields. A field name that resolves to *no* column at all
/// is deliberately NOT cached: field names come from the caller (and under the
/// quickwit `_dynamic` field every string resolves), so memoizing misses would
/// let an unbounded set of arbitrary names accumulate.
///
/// Memory is therefore bounded by the number of distinct (field, column type)
/// pairs that actually exist and are queried on the segment. Those columns are
/// kept alive for the lifetime of the `SegmentReader`, which is exactly the
/// lifetime a long-lived searcher wants them for.
///
/// The maps are keyed by the *user-supplied* field name (not the resolved column
/// name) so that a hit also skips `resolve_field`. They are nested rather than
/// keyed on a `(String, ColumnType)` tuple so lookups can borrow the name and
/// avoid allocating on the hit path.
#[derive(Default)]
struct ColumnCache {
    handles: RwLock<HashMap<String, Arc<Vec<DynamicColumnHandle>>>>,
    opened: RwLock<HashMap<String, HashMap<ColumnType, Option<DynamicColumn>>>>,
    lenient: RwLock<HashMap<String, HashMap<ColumnType, Option<Column<u64>>>>>,
}

/// Provides access to all of the BitpackedFastFieldReader.
///
/// Internally, `FastFieldReaders` have preloaded fast field readers,
/// and just wraps several `HashMap`.
#[derive(Clone)]
pub struct FastFieldReaders {
    columnar: Arc<ColumnarReader>,
    schema: Schema,
    /// Shared with every clone: clones of a `FastFieldReaders` address the same
    /// immutable segment, so they should share the memoized columns rather than
    /// each re-opening them.
    cache: Arc<ColumnCache>,
}

impl FastFieldReaders {
    pub(crate) fn open(fast_field_file: FileSlice, schema: Schema) -> io::Result<FastFieldReaders> {
        let columnar = Arc::new(ColumnarReader::open(fast_field_file)?);
        Ok(FastFieldReaders {
            columnar,
            schema,
            cache: Arc::new(ColumnCache::default()),
        })
    }

    /// Column handles for `field_name`, memoized (step 1 above).
    ///
    /// A field that resolves to no column is *not* cached (see [`ColumnCache`]),
    /// and an *invalid* field (e.g. not configured as fast) keeps returning its
    /// error uncached, since that is a caller bug rather than a property of the
    /// segment.
    fn cached_column_handles(
        &self,
        field_name: &str,
    ) -> crate::Result<Arc<Vec<DynamicColumnHandle>>> {
        if let Some(handles) = self
            .cache
            .handles
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(field_name)
        {
            return Ok(Arc::clone(handles));
        }
        let handles: Vec<DynamicColumnHandle> = match self.resolve_field(field_name)? {
            Some(resolved_field_name) => self.columnar.read_columns(&resolved_field_name)?,
            None => Vec::new(),
        };
        let handles = Arc::new(handles);
        // Only names that actually map to a column are cached. Field names come
        // from the caller (and with quickwit's `_dynamic` field *every* string
        // resolves), so caching misses would let an unbounded set of arbitrary
        // names accumulate for the lifetime of the segment reader. Columns that
        // exist are bounded by the segment itself.
        if !handles.is_empty() {
            self.cache
                .handles
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(field_name.to_string(), Arc::clone(&handles));
        }
        Ok(handles)
    }

    /// The opened column of type `column_type` for `field_name`, memoized
    /// (step 2 above).
    fn cached_dynamic_column(
        &self,
        field_name: &str,
        column_type: ColumnType,
    ) -> crate::Result<Option<DynamicColumn>> {
        if let Some(column) = self
            .cache
            .opened
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(field_name)
            .and_then(|by_type| by_type.get(&column_type))
        {
            return Ok(column.clone());
        }
        let handles = self.cached_column_handles(field_name)?;
        if handles.is_empty() {
            // Unknown field: nothing to memoize, and caching it would let
            // arbitrary caller-supplied names accumulate (see
            // `cached_column_handles`).
            return Ok(None);
        }
        let column_opt: Option<DynamicColumn> = handles
            .iter()
            .find(|column| column.column_type() == column_type)
            .map(|column| column.open())
            .transpose()?;
        // A `None` here means "this field has no column of this type" — bounded
        // by the segment's own fields, so it is worth remembering.
        self.cache
            .opened
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(field_name.to_string())
            .or_default()
            .insert(column_type, column_opt.clone());
        Ok(column_opt)
    }

    /// The `u64`-lenient view of the `column_type` column of `field_name`,
    /// memoized (step 3 above). This is the view range queries run on.
    fn cached_u64_lenient_column(
        &self,
        field_name: &str,
        handle: &DynamicColumnHandle,
    ) -> crate::Result<Option<Column<u64>>> {
        let column_type = handle.column_type();
        if let Some(column) = self
            .cache
            .lenient
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(field_name)
            .and_then(|by_type| by_type.get(&column_type))
        {
            return Ok(column.clone());
        }
        let column_opt = handle.open_u64_lenient()?;
        self.cache
            .lenient
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(field_name.to_string())
            .or_default()
            .insert(column_type, column_opt.clone());
        Ok(column_opt)
    }

    fn resolve_field(&self, column_name: &str) -> crate::Result<Option<String>> {
        let default_field_opt: Option<Field> = if cfg!(feature = "quickwit") {
            self.schema.get_field("_dynamic").ok()
        } else {
            None
        };
        self.resolve_column_name_given_default_field(column_name, default_field_opt)
    }

    pub(crate) fn space_usage(&self) -> io::Result<PerFieldSpaceUsage> {
        let mut per_field_usages: Vec<FieldUsage> = Default::default();
        for (mut field_name, column_handle) in self.columnar.iter_columns()? {
            json_path_sep_to_dot(&mut field_name);
            let space_usage = column_handle.space_usage()?;
            let mut field_usage = FieldUsage::empty(field_name);
            field_usage.set_column_usage(space_usage);
            per_field_usages.push(field_usage);
        }
        Ok(PerFieldSpaceUsage::new(per_field_usages))
    }

    pub(crate) fn columnar(&self) -> &ColumnarReader {
        self.columnar.as_ref()
    }

    /// Transforms a user-supplied fast field name into a column name.
    ///
    /// A user-supplied fast field name is not necessarily a schema field name
    /// because we handle fast fields.
    ///
    /// For instance, if the documents look like `{.., "attributes": {"color": "red"}}` and
    /// `attributes` is a json fast field,  a user could want to run a term aggregation over
    /// colors, by referring to the field as `attributes.color`.
    ///
    /// This function transforms `attributes.color` into a column key to be used in the `columnar`.
    ///
    /// The logic works as follows, first we identify which field is targeted by calling
    /// `schema.find_field(..)`. This method will attempt to split the user splied fast field
    /// name by non-escaped dots, and find the longest matching schema field name.
    /// In our case, it would return the (attribute_field, "color").
    ///
    /// If no field is found, but a dynamic field is supplied, then we
    /// will simply assume the user is targeting the dynamic field. (This feature is used in
    /// Quickwit.)
    ///
    /// We then encode the `(field, path)` into the right `columnar_key`.
    fn resolve_column_name_given_default_field<'a>(
        &'a self,
        field_name: &'a str,
        default_field_opt: Option<Field>,
    ) -> crate::Result<Option<String>> {
        let Some((field, path)): Option<(Field, &str)> = self
            .schema
            .find_field_with_default(field_name, default_field_opt)
        else {
            return Ok(None);
        };
        let field_entry: &FieldEntry = self.schema.get_field_entry(field);
        if !field_entry.is_fast() {
            return Err(TantivyError::InvalidArgument(format!(
                "Field {field_name:?} is not configured as fast field"
            )));
        }
        Ok(match (field_entry.field_type(), path) {
            (FieldType::JsonObject(json_options), path) if !path.is_empty() => {
                Some(encode_column_name(
                    field_entry.name(),
                    path,
                    json_options.is_expand_dots_enabled(),
                ))
            }
            (_, "") => Some(field_entry.name().to_string()),
            _ => None,
        })
    }

    /// Returns a typed column associated to a given field name.
    ///
    /// If no column associated with that field_name exists,
    /// or existing columns do not have the required type,
    /// returns `None`.
    pub fn column_opt<T>(&self, field_name: &str) -> crate::Result<Option<Column<T>>>
    where
        T: HasAssociatedColumnType,
        DynamicColumn: Into<Option<Column<T>>>,
    {
        let Some(dynamic_column) = self.cached_dynamic_column(field_name, T::column_type())? else {
            return Ok(None);
        };
        Ok(dynamic_column.into())
    }

    /// Returns the number of `bytes` associated with a column.
    ///
    /// Returns 0 if the column does not exist.
    pub fn column_num_bytes(&self, field: &str) -> crate::Result<ByteCount> {
        Ok(self
            .cached_column_handles(field)?
            .iter()
            .map(|column_handle| column_handle.num_bytes())
            .sum())
    }

    /// Returns a typed column value object.
    ///
    /// In that column value:
    /// - Rows with no value are associated with the default value.
    /// - Rows with several values are associated with the first value.
    pub fn column_first_or_default<T>(&self, field: &str) -> crate::Result<Arc<dyn ColumnValues<T>>>
    where
        T: PartialOrd + Copy + HasAssociatedColumnType + Send + Sync + 'static,
        DynamicColumn: Into<Option<Column<T>>>,
    {
        let col: Column<T> = self.column(field)?;
        Ok(col.first_or_default_col(T::default_value()))
    }

    /// Returns a typed column associated to a given field name.
    ///
    /// Returns an error if no column associated with that field_name exists.
    fn column<T>(&self, field: &str) -> crate::Result<Column<T>>
    where
        T: PartialOrd + Copy + HasAssociatedColumnType + Send + Sync + 'static,
        DynamicColumn: Into<Option<Column<T>>>,
    {
        let col_opt: Option<Column<T>> = self.column_opt(field)?;
        col_opt.ok_or_else(|| {
            crate::TantivyError::SchemaError(format!(
                "Field `{field}` is missing or is not configured as a fast field."
            ))
        })
    }

    /// Returns the `u64` fast field reader reader associated with `field`.
    ///
    /// If `field` is not a u64 fast field, this method returns an Error.
    pub fn u64(&self, field: &str) -> crate::Result<Column<u64>> {
        self.column(field)
    }

    /// Returns the `date` fast field reader reader associated with `field`.
    ///
    /// If `field` is not a date fast field, this method returns an Error.
    pub fn date(&self, field: &str) -> crate::Result<Column<common::DateTime>> {
        self.column(field)
    }

    /// Returns the `ip` fast field reader reader associated to `field`.
    ///
    /// If `field` is not a u128 fast field, this method returns an Error.
    pub fn ip_addr(&self, field: &str) -> crate::Result<Column<Ipv6Addr>> {
        self.column(field)
    }

    /// Returns a `str` column.
    pub fn str(&self, field_name: &str) -> crate::Result<Option<StrColumn>> {
        let Some(dynamic_column) = self.cached_dynamic_column(field_name, ColumnType::Str)? else {
            return Ok(None);
        };
        Ok(dynamic_column.into())
    }

    /// Returns a `bytes` column.
    pub fn bytes(&self, field_name: &str) -> crate::Result<Option<BytesColumn>> {
        let Some(dynamic_column) = self.cached_dynamic_column(field_name, ColumnType::Bytes)? else {
            return Ok(None);
        };
        Ok(dynamic_column.into())
    }

    /// Returns a `dynamic_column_handle`.
    pub fn dynamic_column_handle(
        &self,
        field_name: &str,
        column_type: ColumnType,
    ) -> crate::Result<Option<DynamicColumnHandle>> {
        let dynamic_column_handle_opt = self
            .cached_column_handles(field_name)?
            .iter()
            .find(|column| column.column_type() == column_type)
            .cloned();
        Ok(dynamic_column_handle_opt)
    }

    /// Returns all `dynamic_column_handle` that match the given field name.
    pub fn dynamic_column_handles(
        &self,
        field_name: &str,
    ) -> crate::Result<Vec<DynamicColumnHandle>> {
        Ok(self.cached_column_handles(field_name)?.as_ref().clone())
    }

    /// Returns all `dynamic_column_handle` that are inner fields of the provided JSON path.
    pub fn dynamic_subpath_column_handles(
        &self,
        root_path: &str,
    ) -> crate::Result<Vec<DynamicColumnHandle>> {
        let Some(resolved_field_name) = self.resolve_field(root_path)? else {
            return Ok(Vec::new());
        };
        let dynamic_column_handles = self
            .columnar
            .read_subpath_columns(&resolved_field_name)?
            .into_iter()
            .collect();
        Ok(dynamic_column_handles)
    }

    #[doc(hidden)]
    pub async fn list_dynamic_column_handles(
        &self,
        field_name: &str,
    ) -> crate::Result<Vec<DynamicColumnHandle>> {
        let Some(resolved_field_name) = self.resolve_field(field_name)? else {
            return Ok(Vec::new());
        };
        let columns = self
            .columnar
            .read_columns_async(&resolved_field_name)
            .await?;
        Ok(columns)
    }

    #[doc(hidden)]
    pub async fn list_subpath_dynamic_column_handles(
        &self,
        root_path: &str,
    ) -> crate::Result<Vec<DynamicColumnHandle>> {
        let Some(resolved_field_name) = self.resolve_field(root_path)? else {
            return Ok(Vec::new());
        };
        let columns = self
            .columnar
            .read_subpath_columns_async(&resolved_field_name)
            .await?;
        Ok(columns)
    }

    /// Returns the `u64` column used to represent any `u64`-mapped typed (String/Bytes term ids,
    /// i64, u64, f64, DateTime).
    ///
    /// Returns Ok(None) for empty columns
    #[doc(hidden)]
    pub fn u64_lenient_for_type(
        &self,
        type_white_list_opt: Option<&[ColumnType]>,
        field_name: &str,
    ) -> crate::Result<Option<(Column<u64>, ColumnType)>> {
        for col in self.cached_column_handles(field_name)?.iter() {
            if let Some(type_white_list) = type_white_list_opt {
                if !type_white_list.contains(&col.column_type()) {
                    continue;
                }
            }
            // Only whitelisted columns are opened, exactly as before — the cache
            // is consulted per column, so filtering semantics are unchanged.
            if let Some(col_u64) = self.cached_u64_lenient_column(field_name, col)? {
                return Ok(Some((col_u64, col.column_type())));
            }
        }
        Ok(None)
    }

    /// Returns the all `u64` column used to represent any `u64`-mapped typed (String/Bytes term
    /// ids, i64, u64, f64, bool, DateTime).
    ///
    /// In case of JSON, there may be two columns. One for term and one for numerical types. (This
    /// may change later to 3 types if JSON handles DateTime)
    #[doc(hidden)]
    pub fn u64_lenient_for_type_all(
        &self,
        type_white_list_opt: Option<&[ColumnType]>,
        field_name: &str,
    ) -> crate::Result<Vec<(Column<u64>, ColumnType)>> {
        let mut columns_and_types = Vec::new();
        for col in self.cached_column_handles(field_name)?.iter() {
            if let Some(type_white_list) = type_white_list_opt {
                if !type_white_list.contains(&col.column_type()) {
                    continue;
                }
            }
            if let Some(col_u64) = self.cached_u64_lenient_column(field_name, col)? {
                columns_and_types.push((col_u64, col.column_type()));
            }
        }
        Ok(columns_and_types)
    }

    /// Returns the `u64` column used to represent any `u64`-mapped typed (i64, u64, f64, DateTime).
    ///
    /// Returns Ok(None) for empty columns
    #[doc(hidden)]
    pub fn u64_lenient(
        &self,
        field_name: &str,
    ) -> crate::Result<Option<(Column<u64>, ColumnType)>> {
        self.u64_lenient_for_type(None, field_name)
    }

    /// Returns the `i64` fast field reader reader associated with `field`.
    ///
    /// If `field` is not a i64 fast field, this method returns an Error.
    pub fn i64(&self, field_name: &str) -> crate::Result<Column<i64>> {
        self.column(field_name)
    }

    /// Returns the `f64` fast field reader reader associated with `field`.
    ///
    /// If `field` is not a f64 fast field, this method returns an Error.
    pub fn f64(&self, field_name: &str) -> crate::Result<Column<f64>> {
        self.column(field_name)
    }

    /// Returns the `bool` fast field reader reader associated with `field`.
    ///
    /// If `field` is not a bool fast field, this method returns an Error.
    pub fn bool(&self, field_name: &str) -> crate::Result<Column<bool>> {
        self.column(field_name)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use columnar::ColumnType;

    use crate::schema::{JsonObjectOptions, Schema, FAST};
    use crate::tokenizer::RAW_TOKENIZER_NAME;
    use crate::{Index, IndexWriter, TantivyDocument};

    #[test]
    fn test_fast_field_reader_resolve_with_dynamic_internal() {
        let mut schema_builder = Schema::builder();
        schema_builder.add_i64_field("age", FAST);
        schema_builder.add_json_field("json_expand_dots_disabled", FAST);
        schema_builder.add_json_field(
            "json_expand_dots_enabled",
            JsonObjectOptions::default()
                .set_fast(RAW_TOKENIZER_NAME)
                .set_expand_dots_enabled(),
        );
        let dynamic_field = schema_builder.add_json_field("_dyna", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer
            .add_document(TantivyDocument::default())
            .unwrap();
        index_writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let reader = searcher.segment_reader(0u32);
        let fast_field_readers = reader.fast_fields();
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field("age", None)
                .unwrap(),
            Some("age".to_string())
        );
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field("age", Some(dynamic_field))
                .unwrap(),
            Some("age".to_string())
        );
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field(
                    "json_expand_dots_disabled.attr.color",
                    None
                )
                .unwrap(),
            Some("json_expand_dots_disabled\u{1}attr\u{1}color".to_string())
        );
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field(
                    "json_expand_dots_disabled.attr\\.color",
                    Some(dynamic_field)
                )
                .unwrap(),
            Some("json_expand_dots_disabled\u{1}attr.color".to_string())
        );
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field(
                    "json_expand_dots_enabled.attr\\.color",
                    Some(dynamic_field)
                )
                .unwrap(),
            Some("json_expand_dots_enabled\u{1}attr\u{1}color".to_string())
        );
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field("notinschema.attr.color", None)
                .unwrap(),
            None
        );
        assert_eq!(
            fast_field_readers
                .resolve_column_name_given_default_field(
                    "notinschema.attr.color",
                    Some(dynamic_field)
                )
                .unwrap(),
            Some("_dyna\u{1}notinschema\u{1}attr\u{1}color".to_string())
        );
    }

    #[test]
    fn test_fast_field_reader_dynamic_column_handles() {
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_u64_field("id", FAST);
        let json = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer
            .add_document(doc!(id=> 1u64, json => json!({"foo": 42})))
            .unwrap();
        index_writer
            .add_document(doc!(id=> 2u64, json => json!({"foo": true})))
            .unwrap();
        index_writer
            .add_document(doc!(id=> 3u64, json => json!({"foo": "bar"})))
            .unwrap();
        index_writer.commit().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let reader = searcher.segment_reader(0u32);
        let fast_fields = reader.fast_fields();
        let id_columns = fast_fields.dynamic_column_handles("id").unwrap();
        assert_eq!(id_columns.len(), 1);
        assert_eq!(id_columns.first().unwrap().column_type(), ColumnType::U64);

        let foo_columns = fast_fields.dynamic_column_handles("json.foo").unwrap();
        assert_eq!(foo_columns.len(), 3);
        assert!(foo_columns
            .iter()
            .any(|column| column.column_type() == ColumnType::I64));
        assert!(foo_columns
            .iter()
            .any(|column| column.column_type() == ColumnType::Bool));
        assert!(foo_columns
            .iter()
            .any(|column| column.column_type() == ColumnType::Str));

        let json_columns = fast_fields.dynamic_column_handles("json").unwrap();
        assert_eq!(json_columns.len(), 0);

        let json_subcolumns = fast_fields.dynamic_subpath_column_handles("json").unwrap();
        assert_eq!(json_subcolumns.len(), 3);

        let foo_subcolumns = fast_fields
            .dynamic_subpath_column_handles("json.foo")
            .unwrap();
        assert_eq!(foo_subcolumns.len(), 0);
    }

    /// A segment with one u64, one f64 and a polymorphic json fast field —
    /// mirrors the shape the column cache has to serve (typed lookups plus the
    /// `u64`-lenient views that range queries use).
    fn cache_test_index() -> Index {
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_u64_field("id", FAST);
        let id2 = schema_builder.add_u64_field("id2", FAST);
        let score = schema_builder.add_f64_field("score", FAST);
        let json = schema_builder.add_json_field("json", FAST);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer
            .add_document(
                doc!(id => 1u64, id2 => 10u64, score => 1.5f64, json => json!({"foo": 42})),
            )
            .unwrap();
        index_writer
            .add_document(
                doc!(id => 2u64, id2 => 20u64, score => 2.5f64, json => json!({"foo": "bar"})),
            )
            .unwrap();
        index_writer.commit().unwrap();
        index
    }

    #[test]
    fn test_column_cache_memoizes_opened_columns() {
        // The point of the cache: opening the same column twice must not
        // re-parse it. Identity (not just equality) of the underlying values
        // is what proves the second call was served from the cache.
        let index = cache_test_index();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        let first = fast_fields.u64("id").unwrap();
        let second = fast_fields.u64("id").unwrap();
        assert!(
            Arc::ptr_eq(&first.values, &second.values),
            "repeated u64 opens must share one parsed column"
        );

        let first_f64 = fast_fields.f64("score").unwrap();
        let second_f64 = fast_fields.f64("score").unwrap();
        assert!(
            Arc::ptr_eq(&first_f64.values, &second_f64.values),
            "repeated f64 opens must share one parsed column"
        );

        // Distinct fields of the same type must stay distinct — the cache is
        // keyed per field, not a single shared slot.
        let other_u64 = fast_fields.u64("id2").unwrap();
        assert!(
            !Arc::ptr_eq(&first.values, &other_u64.values),
            "different fields must not collide in the cache"
        );
    }

    #[test]
    fn test_column_cache_preserves_values() {
        // Caching must not change what the columns read back.
        let index = cache_test_index();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        for _ in 0..3 {
            let ids = fast_fields.u64("id").unwrap();
            assert_eq!(ids.first(0).unwrap(), 1u64);
            assert_eq!(ids.first(1).unwrap(), 2u64);
            let scores = fast_fields.f64("score").unwrap();
            assert_eq!(scores.first(0).unwrap(), 1.5f64);
            assert_eq!(scores.first(1).unwrap(), 2.5f64);
        }
    }

    #[test]
    fn test_column_cache_memoizes_u64_lenient_columns() {
        // The path range queries take (`u64_lenient_for_type`) is cached too —
        // this was the dominant cost in the geocode profile.
        let index = cache_test_index();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        let (first, first_type) = fast_fields.u64_lenient_for_type(None, "id").unwrap().unwrap();
        let (second, second_type) = fast_fields.u64_lenient_for_type(None, "id").unwrap().unwrap();
        assert_eq!(first_type, ColumnType::U64);
        assert_eq!(second_type, ColumnType::U64);
        assert!(
            Arc::ptr_eq(&first.values, &second.values),
            "repeated lenient opens must share one parsed column"
        );
    }

    #[test]
    fn test_column_cache_preserves_type_whitelist_semantics() {
        // A json field carries several typed columns for one path. The cache is
        // consulted per column, so a whitelist must still select exactly the
        // requested type — and never open a non-whitelisted one.
        let index = cache_test_index();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        for _ in 0..2 {
            let (_, col_type) = fast_fields
                .u64_lenient_for_type(Some(&[ColumnType::Str]), "json.foo")
                .unwrap()
                .expect("json.foo has a str column");
            assert_eq!(col_type, ColumnType::Str);

            let (_, col_type) = fast_fields
                .u64_lenient_for_type(Some(&[ColumnType::I64]), "json.foo")
                .unwrap()
                .expect("json.foo has an i64 column");
            assert_eq!(col_type, ColumnType::I64);

            assert!(
                fast_fields
                    .u64_lenient_for_type(Some(&[ColumnType::Bool]), "json.foo")
                    .unwrap()
                    .is_none(),
                "no bool column was indexed for json.foo"
            );
        }
    }

    #[test]
    fn test_column_cache_caches_missing_columns() {
        // Negative lookups are cached as `None`; they must keep reporting
        // absence rather than erroring or resurrecting a column.
        let index = cache_test_index();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        for _ in 0..3 {
            assert!(fast_fields
                .column_opt::<u64>("does_not_exist")
                .unwrap()
                .is_none());
            // Present field, wrong type: also a cached negative.
            assert!(fast_fields.column_opt::<i64>("id").unwrap().is_none());
        }
        // ...and the real column is still reachable afterwards.
        assert_eq!(fast_fields.u64("id").unwrap().first(0).unwrap(), 1u64);
    }

    #[test]
    fn test_column_cache_does_not_cache_errors() {
        // A field that exists in the schema but is not FAST is a caller error,
        // not a property of the segment: it must keep erroring on every call
        // rather than being memoized (as a `None` or otherwise).
        let mut schema_builder = Schema::builder();
        let id = schema_builder.add_u64_field("id", FAST);
        let text = schema_builder.add_text_field("text", crate::schema::TEXT);
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer_for_tests().unwrap();
        index_writer
            .add_document(doc!(id => 1u64, text => "hello"))
            .unwrap();
        index_writer.commit().unwrap();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        for _ in 0..3 {
            assert!(
                fast_fields.column_opt::<u64>("text").is_err(),
                "a non-fast field must keep reporting the error"
            );
        }
        assert_eq!(fast_fields.u64("id").unwrap().first(0).unwrap(), 1u64);
    }

    #[test]
    fn test_column_cache_column_num_bytes() {
        // `column_num_bytes` was rewired onto the handle cache; a present field
        // still reports its size and an absent one still reports 0.
        let index = cache_test_index();
        let searcher = index.reader().unwrap().searcher();
        let fast_fields = searcher.segment_reader(0u32).fast_fields();

        for _ in 0..2 {
            assert!(fast_fields.column_num_bytes("id").unwrap().get_bytes() > 0);
            assert_eq!(
                fast_fields
                    .column_num_bytes("does_not_exist")
                    .unwrap()
                    .get_bytes(),
                0u64
            );
        }
    }
}
