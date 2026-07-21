use crate::rms_norm::CudaBackend;
use crate::runtime::{loom_status_result, DeviceBuffer};
use crate::CudaExecutorError;
use half::{bf16, f16};
use loom_kernels::{DType, RopePagedKvWriteSpec, RotaryStyle};

impl CudaBackend {
    /// Applies F32 RoPE in place and writes K/V to contiguous NHD caches.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_paged_kv_write_f32(
        &self,
        query: &mut DeviceBuffer<f32>,
        key: &mut DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<i64>,
        cos_sin_cache: &DeviceBuffer<f32>,
        key_cache: &mut DeviceBuffer<f32>,
        value_cache: &mut DeviceBuffer<f32>,
        slot_mapping: &DeviceBuffer<i64>,
        spec: RopePagedKvWriteSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let shape = validate_buffers(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            slot_mapping,
            spec,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rope_paged_kv_write_f32(
                query.as_mut_ptr(),
                key.as_mut_ptr(),
                value.as_ptr(),
                positions.as_ptr(),
                cos_sin_cache.as_ptr(),
                key_cache.as_mut_ptr(),
                value_cache.as_mut_ptr(),
                slot_mapping.as_ptr(),
                shape.tokens,
                shape.cache_tokens,
                shape.query_heads,
                shape.kv_heads,
                shape.head_size,
                shape.value_head_size,
                shape.rotary_dim,
                shape.max_position,
                shape.num_blocks,
                shape.block_size,
                shape.query_token_stride,
                shape.query_head_stride,
                shape.key_token_stride,
                shape.source_key_head_stride,
                shape.value_token_stride,
                shape.source_value_head_stride,
                shape.key_block_stride,
                shape.key_page_stride,
                shape.key_head_stride,
                shape.value_block_stride,
                shape.value_page_stride,
                shape.value_head_stride,
                shape.is_neox,
                self.stream().raw(),
            )
        })
    }

    /// Applies FP16 RoPE in place and writes K/V to contiguous NHD caches.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_paged_kv_write_f16(
        &self,
        query: &mut DeviceBuffer<f16>,
        key: &mut DeviceBuffer<f16>,
        value: &DeviceBuffer<f16>,
        positions: &DeviceBuffer<i64>,
        cos_sin_cache: &DeviceBuffer<f16>,
        key_cache: &mut DeviceBuffer<f16>,
        value_cache: &mut DeviceBuffer<f16>,
        slot_mapping: &DeviceBuffer<i64>,
        spec: RopePagedKvWriteSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let shape = validate_buffers(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            slot_mapping,
            spec,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rope_paged_kv_write_f16(
                query.as_mut_ptr().cast::<u16>(),
                key.as_mut_ptr().cast::<u16>(),
                value.as_ptr().cast::<u16>(),
                positions.as_ptr(),
                cos_sin_cache.as_ptr().cast::<u16>(),
                key_cache.as_mut_ptr().cast::<u16>(),
                value_cache.as_mut_ptr().cast::<u16>(),
                slot_mapping.as_ptr(),
                shape.tokens,
                shape.cache_tokens,
                shape.query_heads,
                shape.kv_heads,
                shape.head_size,
                shape.value_head_size,
                shape.rotary_dim,
                shape.max_position,
                shape.num_blocks,
                shape.block_size,
                shape.query_token_stride,
                shape.query_head_stride,
                shape.key_token_stride,
                shape.source_key_head_stride,
                shape.value_token_stride,
                shape.source_value_head_stride,
                shape.key_block_stride,
                shape.key_page_stride,
                shape.key_head_stride,
                shape.value_block_stride,
                shape.value_page_stride,
                shape.value_head_stride,
                shape.is_neox,
                self.stream().raw(),
            )
        })
    }

    /// Applies BF16 RoPE in place and writes K/V to contiguous NHD caches.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_paged_kv_write_bf16(
        &self,
        query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>,
        value: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<i64>,
        cos_sin_cache: &DeviceBuffer<bf16>,
        key_cache: &mut DeviceBuffer<bf16>,
        value_cache: &mut DeviceBuffer<bf16>,
        slot_mapping: &DeviceBuffer<i64>,
        spec: RopePagedKvWriteSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let shape = validate_buffers(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            slot_mapping,
            spec,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rope_paged_kv_write_bf16(
                query.as_mut_ptr().cast::<u16>(),
                key.as_mut_ptr().cast::<u16>(),
                value.as_ptr().cast::<u16>(),
                positions.as_ptr(),
                cos_sin_cache.as_ptr().cast::<u16>(),
                key_cache.as_mut_ptr().cast::<u16>(),
                value_cache.as_mut_ptr().cast::<u16>(),
                slot_mapping.as_ptr(),
                shape.tokens,
                shape.cache_tokens,
                shape.query_heads,
                shape.kv_heads,
                shape.head_size,
                shape.value_head_size,
                shape.rotary_dim,
                shape.max_position,
                shape.num_blocks,
                shape.block_size,
                shape.query_token_stride,
                shape.query_head_stride,
                shape.key_token_stride,
                shape.source_key_head_stride,
                shape.value_token_stride,
                shape.source_value_head_stride,
                shape.key_block_stride,
                shape.key_page_stride,
                shape.key_head_stride,
                shape.value_block_stride,
                shape.value_page_stride,
                shape.value_head_stride,
                shape.is_neox,
                self.stream().raw(),
            )
        })
    }
}

fn require_dtype(spec: RopePagedKvWriteSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.rotary().dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "{expected:?} RoPE+paged-KV cannot execute {:?}",
            spec.rotary().dtype()
        )))
    }
}

#[derive(Clone, Copy)]
struct AbiShape {
    tokens: u32,
    cache_tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    rotary_dim: u32,
    max_position: u32,
    num_blocks: u32,
    block_size: u32,
    query_token_stride: u64,
    query_head_stride: u64,
    key_token_stride: u64,
    source_key_head_stride: u64,
    value_token_stride: u64,
    source_value_head_stride: u64,
    key_block_stride: u64,
    key_page_stride: u64,
    key_head_stride: u64,
    value_block_stride: u64,
    value_page_stride: u64,
    value_head_stride: u64,
    is_neox: u32,
}

#[allow(clippy::too_many_arguments)]
fn validate_buffers<T: Copy>(
    query: &DeviceBuffer<T>,
    key: &DeviceBuffer<T>,
    value: &DeviceBuffer<T>,
    positions: &DeviceBuffer<i64>,
    cos_sin_cache: &DeviceBuffer<T>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    slot_mapping: &DeviceBuffer<i64>,
    spec: RopePagedKvWriteSpec,
) -> Result<AbiShape, CudaExecutorError> {
    let rotary = spec.rotary();
    query.require_len(rotary.query_numel(), "RoPE query")?;
    key.require_len(rotary.key_numel(), "RoPE key")?;
    value.require_len(spec.value_numel(), "RoPE value")?;
    positions.require_len(rotary.tokens(), "RoPE positions")?;
    cos_sin_cache.require_len(rotary.cos_sin_cache_numel(), "RoPE cos/sin cache")?;
    key_cache.require_len(spec.key_cache_numel(), "paged key cache")?;
    value_cache.require_len(spec.value_cache_numel(), "paged value cache")?;
    slot_mapping.require_len(rotary.tokens(), "paged slot mapping")?;

    let u32_value = |value: usize, name: &str| {
        u32::try_from(value)
            .map_err(|_| CudaExecutorError::InvalidContract(format!("{name} exceeds the CUDA ABI")))
    };
    let source_key_head_stride = u64::try_from(rotary.head_size())
        .map_err(|_| CudaExecutorError::InvalidContract("key head stride overflow".into()))?;
    let query_head_stride = source_key_head_stride;
    let query_token_stride = u64::try_from(
        rotary
            .query_heads()
            .checked_mul(rotary.head_size())
            .ok_or_else(|| {
                CudaExecutorError::InvalidContract("query token stride overflow".into())
            })?,
    )
    .map_err(|_| CudaExecutorError::InvalidContract("query token stride overflow".into()))?;
    let key_token_stride = u64::try_from(
        rotary
            .key_heads()
            .checked_mul(rotary.head_size())
            .ok_or_else(|| {
                CudaExecutorError::InvalidContract("key token stride overflow".into())
            })?,
    )
    .map_err(|_| CudaExecutorError::InvalidContract("key token stride overflow".into()))?;
    let key_head_stride = u64::try_from(rotary.head_size())
        .map_err(|_| CudaExecutorError::InvalidContract("key head stride overflow".into()))?;
    let key_page_stride = u64::try_from(
        rotary
            .key_heads()
            .checked_mul(rotary.head_size())
            .ok_or_else(|| CudaExecutorError::InvalidContract("key page stride overflow".into()))?,
    )
    .map_err(|_| CudaExecutorError::InvalidContract("key page stride overflow".into()))?;
    let key_block_stride = key_page_stride
        .checked_mul(spec.block_size() as u64)
        .ok_or_else(|| CudaExecutorError::InvalidContract("key block stride overflow".into()))?;
    let source_value_head_stride = u64::try_from(spec.value_head_size())
        .map_err(|_| CudaExecutorError::InvalidContract("value head stride overflow".into()))?;
    let value_token_stride = u64::try_from(
        rotary
            .key_heads()
            .checked_mul(spec.value_head_size())
            .ok_or_else(|| {
                CudaExecutorError::InvalidContract("value token stride overflow".into())
            })?,
    )
    .map_err(|_| CudaExecutorError::InvalidContract("value token stride overflow".into()))?;
    let value_head_stride = u64::try_from(spec.value_head_size())
        .map_err(|_| CudaExecutorError::InvalidContract("value head stride overflow".into()))?;
    let value_page_stride = u64::try_from(
        rotary
            .key_heads()
            .checked_mul(spec.value_head_size())
            .ok_or_else(|| {
                CudaExecutorError::InvalidContract("value page stride overflow".into())
            })?,
    )
    .map_err(|_| CudaExecutorError::InvalidContract("value page stride overflow".into()))?;
    let value_block_stride = value_page_stride
        .checked_mul(spec.block_size() as u64)
        .ok_or_else(|| CudaExecutorError::InvalidContract("value block stride overflow".into()))?;

    Ok(AbiShape {
        tokens: u32_value(rotary.tokens(), "token count")?,
        cache_tokens: u32_value(rotary.tokens(), "cache token count")?,
        query_heads: u32_value(rotary.query_heads(), "query head count")?,
        kv_heads: u32_value(rotary.key_heads(), "KV head count")?,
        head_size: u32_value(rotary.head_size(), "head size")?,
        value_head_size: u32_value(spec.value_head_size(), "value head size")?,
        rotary_dim: u32_value(rotary.rotary_dim(), "rotary dimension")?,
        max_position: u32_value(rotary.max_position(), "maximum position")?,
        num_blocks: u32_value(spec.num_blocks(), "cache block count")?,
        block_size: u32_value(spec.block_size(), "cache block size")?,
        query_token_stride,
        query_head_stride,
        key_token_stride,
        source_key_head_stride,
        value_token_stride,
        source_value_head_stride,
        key_block_stride,
        key_page_stride,
        key_head_stride,
        value_block_stride,
        value_page_stride,
        value_head_stride,
        is_neox: u32::from(rotary.style() == RotaryStyle::NeoX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use loom_kernels::{rope_paged_kv_write_f32_reference, RotaryEmbeddingSpec, RotaryStyle};

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let rotary =
            RotaryEmbeddingSpec::new(2, 2, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
        let spec = RopePagedKvWriteSpec::new(rotary, 4, 1, 4).unwrap();
        let positions = [0_i64, 1];
        let slots = [3_i64, -1];
        let cos_sin = [1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let query = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let key = [17.0_f32, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0];
        let value = [25.0_f32, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0];
        let mut expected_query = query;
        let mut expected_key = key;
        let mut expected_key_cache = [-5.0_f32; 16];
        let mut expected_value_cache = [-5.0_f32; 16];
        rope_paged_kv_write_f32_reference(
            &mut expected_query,
            &mut expected_key,
            &value,
            &positions,
            &cos_sin,
            &mut expected_key_cache,
            &mut expected_value_cache,
            &slots,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let mut query_device = DeviceBuffer::from_slice(&query).unwrap();
        let mut key_device = DeviceBuffer::from_slice(&key).unwrap();
        let value_device = DeviceBuffer::from_slice(&value).unwrap();
        let positions_device = DeviceBuffer::from_slice(&positions).unwrap();
        let cos_sin_device = DeviceBuffer::from_slice(&cos_sin).unwrap();
        let mut key_cache_device = DeviceBuffer::from_slice(&[-5.0_f32; 16]).unwrap();
        let mut value_cache_device = DeviceBuffer::from_slice(&[-5.0_f32; 16]).unwrap();
        let slots_device = DeviceBuffer::from_slice(&slots).unwrap();

        backend
            .rope_paged_kv_write_f32(
                &mut query_device,
                &mut key_device,
                &value_device,
                &positions_device,
                &cos_sin_device,
                &mut key_cache_device,
                &mut value_cache_device,
                &slots_device,
                spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(query_device.copy_to_vec().unwrap(), expected_query);
        assert_eq!(key_device.copy_to_vec().unwrap(), expected_key);
        assert_eq!(key_cache_device.copy_to_vec().unwrap(), expected_key_cache);
        assert_eq!(
            value_cache_device.copy_to_vec().unwrap(),
            expected_value_cache
        );
    }
}
