//! The CUDA runtime as gridoxide-gpu uses it: the generated bindings (cuDSS,
//! the CUDA runtime, and the kernels' launchers in `cuda/kernels.h`), error
//! conversion, and device buffers.

use std::ffi::{CStr, c_void};
use std::ptr;

use crate::uniform::LinearError;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unused_imports,
    unnecessary_transmutes,
    clippy::all
)]
pub(crate) mod ffi {
    include!(concat!(env!("OUT_DIR"), "/cudss_bindings.rs"));
}

pub(crate) fn cuda(err: ffi::cudaError_t, what: &str) -> Result<(), LinearError> {
    if err == ffi::cudaSuccess {
        return Ok(());
    }
    // SAFETY: cudaGetErrorString returns a static, NUL-terminated string for
    // any error code, known or not.
    let msg = unsafe { CStr::from_ptr(ffi::cudaGetErrorString(err)) };
    Err(LinearError::Backend(format!(
        "{what}: {}",
        msg.to_string_lossy()
    )))
}

/// A device allocation of `len` values of `T`, freed on drop.
pub(crate) struct DeviceBuffer<T> {
    pub(crate) ptr: *mut T,
    len: usize,
}

impl<T: Copy> DeviceBuffer<T> {
    pub(crate) fn new(len: usize) -> Result<Self, LinearError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMalloc writes a device pointer to `ptr` or fails.
        cuda(
            unsafe { ffi::cudaMalloc(&mut ptr, len * size_of::<T>()) },
            "cudaMalloc",
        )?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    /// Copies `src` to the device, ordered on `stream`. From pageable memory
    /// the call returns once `src` has been staged, so `src` may be reused.
    pub(crate) fn upload(
        &mut self,
        src: &[T],
        stream: ffi::cudaStream_t,
    ) -> Result<(), LinearError> {
        assert_eq!(src.len(), self.len);
        // SAFETY: both ranges are `len` values long.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    self.ptr.cast(),
                    src.as_ptr().cast(),
                    self.len * size_of::<T>(),
                    ffi::cudaMemcpyHostToDevice,
                    stream,
                )
            },
            "cudaMemcpyAsync (to device)",
        )
    }

    /// Copies the buffer into `dst` once everything queued on `stream`
    /// before it has run.
    pub(crate) fn download(
        &self,
        dst: &mut [T],
        stream: ffi::cudaStream_t,
    ) -> Result<(), LinearError> {
        assert_eq!(dst.len(), self.len);
        // SAFETY: both ranges are `len` values long; the synchronize makes
        // the copy complete before `dst` is read.
        cuda(
            unsafe {
                ffi::cudaMemcpyAsync(
                    dst.as_mut_ptr().cast(),
                    self.ptr.cast(),
                    self.len * size_of::<T>(),
                    ffi::cudaMemcpyDeviceToHost,
                    stream,
                )
            },
            "cudaMemcpyAsync (to host)",
        )?;
        cuda(
            unsafe { ffi::cudaStreamSynchronize(stream) },
            "cudaStreamSynchronize",
        )
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` came from cudaMalloc and is freed once.
            unsafe { ffi::cudaFree(self.ptr.cast()) };
        }
    }
}

impl<T> DeviceBuffer<T> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// Copies `src` to `dst` on the device, ordered on `stream`, as
/// [`DeviceBuffer::upload`] does into memory it does not own.
///
/// # Safety
///
/// `dst` must be a device allocation of at least `src.len()` values.
pub(crate) unsafe fn copy_to_device<T: Copy>(
    dst: *mut T,
    src: &[T],
    stream: ffi::cudaStream_t,
) -> Result<(), LinearError> {
    // SAFETY: the caller's.
    cuda(
        unsafe {
            ffi::cudaMemcpyAsync(
                dst.cast(),
                src.as_ptr().cast(),
                std::mem::size_of_val(src),
                ffi::cudaMemcpyHostToDevice,
                stream,
            )
        },
        "cudaMemcpyAsync (to device)",
    )
}

/// Page-locked host memory for `len` values of `T`, freed on drop: what a
/// device-to-host copy can run into at the bus's speed, where a copy into
/// pageable memory is staged by the driver (1.3 GB/s measured on the A100
/// machine, into freshly allocated memory).
pub(crate) struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> PinnedBuffer<T> {
    pub(crate) fn new(len: usize) -> Result<Self, LinearError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        // SAFETY: cudaMallocHost writes a host pointer to `ptr` or fails.
        cuda(
            unsafe { ffi::cudaMallocHost(&mut ptr, len.max(1) * size_of::<T>()) },
            "cudaMallocHost",
        )?;
        Ok(Self {
            ptr: ptr.cast(),
            len,
        })
    }

    /// The allocation, for a copy to write into. Through `&self`: a session
    /// writes one staging buffer by DMA while a host thread reads the other.
    pub(crate) fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// The first `len` values. Only once a copy into them has completed.
    pub(crate) fn prefix(&self, len: usize) -> &[T] {
        assert!(len <= self.len);
        // SAFETY: `ptr` holds `self.len` values of `T`, allocated in `new`;
        // the caller reads them after the copy that wrote them.
        unsafe { std::slice::from_raw_parts(self.ptr, len) }
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from cudaMallocHost and is freed once.
        unsafe { ffi::cudaFreeHost(self.ptr.cast()) };
    }
}

// SAFETY: plain host memory, owned. Shared, it is read through `prefix`
// and written only by copies the owner orders against those reads.
unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

/// A CUDA stream, synchronized and destroyed on drop. Declared before the
/// buffers in a struct that owns both, it is dropped first, so no queued
/// work outlives the memory it uses.
pub(crate) struct Stream(pub(crate) ffi::cudaStream_t);

impl Stream {
    pub(crate) fn new() -> Result<Self, LinearError> {
        let mut stream: ffi::cudaStream_t = ptr::null_mut();
        // SAFETY: cudaStreamCreate writes a stream handle or fails.
        cuda(
            unsafe { ffi::cudaStreamCreate(&mut stream) },
            "cudaStreamCreate",
        )?;
        Ok(Self(stream))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the stream was created in `new` and is destroyed once.
        unsafe {
            ffi::cudaStreamSynchronize(self.0);
            ffi::cudaStreamDestroy(self.0);
        }
    }
}

/// A CUDA event without timing, for ordering one stream after another;
/// destroyed on drop.
pub(crate) struct Event(pub(crate) ffi::cudaEvent_t);

impl Event {
    pub(crate) fn new() -> Result<Self, LinearError> {
        let mut event: ffi::cudaEvent_t = ptr::null_mut();
        // SAFETY: cudaEventCreateWithFlags writes an event handle or fails.
        cuda(
            unsafe { ffi::cudaEventCreateWithFlags(&mut event, ffi::cudaEventDisableTiming) },
            "cudaEventCreateWithFlags",
        )?;
        Ok(Self(event))
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: created in `new`, destroyed once.
        unsafe { ffi::cudaEventDestroy(self.0) };
    }
}

// SAFETY: a CUDA event handle may be used from any host thread.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}
