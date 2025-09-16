using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.UnityIntegration;
using Unity.InferenceEngine;
using UnityEngine;

namespace InferenceEngineWithOpenCVForUnity
{
    /// <summary>
    /// Base class for asynchronous inference processing using UnityInferenceEngine.
    /// Controls Initialize, Dispose, and inference cancellation processing.
    /// </summary>
    /// <typeparam name="TResult">The type of inference result</typeparam>
    public abstract class InferenceEngineManager<TResult> : MonoBehaviour
        where TResult : struct
    {
        #region Fields

        // 1. Constants (if any)

        // 2. Static fields (if any)

        // 3. Serialized instance fields
        [SerializeField]
        private BackendType _backendType = BackendType.CPU;

        // 4. Instance fields
        private CancellationTokenSource _cancellationTokenSource;
        private bool _hasInitDone = false;
        private bool _isInitializing = false;
        private bool _isInferring = false;
        private bool _isDisposing = false;
        private TaskCompletionSource<bool> _initiallizeTaskCompletion = null;
        private TaskCompletionSource<bool> _inferenceTaskCompletion = null;
        private TaskCompletionSource<bool> _disposeTaskCompletion = null;
        private RenderTexture _renderTexture = null;
        private GraphicsBuffer _graphicsBuffer = null;

        // 5. Protected fields
        protected event Action OnInitialize;

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets the backend type for inference engine.
        /// </summary>
        public BackendType BackendType
        {
            get => _backendType;
            set => _backendType = value;
        }

        #endregion

        #region Unity Lifecycle Methods

        /// <summary>
        /// Called when the object is destroyed.
        /// </summary>
        public virtual async Awaitable OnDestroy()
        {
            //Debug.Log("InferenceEngineHandDetector OnDestroy");

            //Dispose();

            await DisposeAsync();

            // TaskCompletionSource objects cleanup
            _initiallizeTaskCompletion?.TrySetCanceled();
            _initiallizeTaskCompletion = null;
            _inferenceTaskCompletion?.TrySetCanceled();
            _inferenceTaskCompletion = null;
            _disposeTaskCompletion?.TrySetCanceled();
            _disposeTaskCompletion = null;
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the inference engine synchronously.
        /// </summary>
        public virtual void Initialize()
        {
            //Debug.LogWarning("Initialize() called");

            // Skip if already initialized
            if (_hasInitDone)
            {
                Debug.LogWarning("Initialize() is already done.");
                return;
            }

            // Skip if currently initializing
            if (_isInitializing)
            {
                Debug.LogWarning("Initialize() is already running.");
                return;
            }

            // Skip if disposing (Initialize() will be disposed anyway)
            if (_isDisposing)
            {
                Debug.LogWarning("isDisposing is true. Initialize() is not called.");
                return;
            }

            _isInitializing = true;
            _initiallizeTaskCompletion = new TaskCompletionSource<bool>(); // Create new task here

            try
            {
                // Execute event if set
                OnInitialize?.Invoke();

                InitializeBase();
                InitializeCustom();

                _hasInitDone = true;

                _isInitializing = false;
                _initiallizeTaskCompletion.TrySetResult(true); // Notify completion to `DisposeAsync()`
            }
            catch (Exception ex)
            {
                // Cleanup state on exception
                _isInitializing = false;
                _initiallizeTaskCompletion?.TrySetException(ex);
                throw; // Re-throw exception
            }
        }

        /// <summary>
        /// Initialize the inference engine asynchronously.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Awaitable task</returns>
        public virtual async Awaitable InitializeAsync(CancellationToken cancellationToken = default)
        {
            //Debug.LogWarning("InitializeAsync() called");

            // Skip if currently initializing to avoid double processing
            if (_isInitializing)
            {
                Debug.LogWarning("InitializeAsync() is already running.");
                return;
            }

            _isInitializing = true;
            _initiallizeTaskCompletion = new TaskCompletionSource<bool>(); // Create new task

            try
            {
                // Wait for DisposeAsync() to complete if disposing
                if (_isDisposing)
                {
                    Debug.LogWarning("InitializeAsync() is waiting for DisposeAsync() to complete.");
                    await _disposeTaskCompletion.Task; // Wait for `DisposeAsync()` completion
                }

                // Call DisposeAsync() if already initialized
                if (_hasInitDone)
                {
                    Debug.Log("Already initialized. Disposing existing resources before re-initialization.");
                    await DisposeAsync();
                }

                // Check for cancellation
                cancellationToken.ThrowIfCancellationRequested();

                // Execute event if set
                OnInitialize?.Invoke();

                InitializeBase();
                InitializeAsyncCustom(cancellationToken);

                _hasInitDone = true;

                _isInitializing = false;
                _initiallizeTaskCompletion.TrySetResult(true); // Notify completion to `DisposeAsync()`

                //Debug.LogWarning("InitializeAsync() finished");
            }
            catch (OperationCanceledException)
            {
                // Cleanup state on cancellation
                _isInitializing = false;
                _initiallizeTaskCompletion.TrySetCanceled();
                throw; // Re-throw exception
            }
            catch (Exception ex)
            {
                // Cleanup state on other exceptions
                _isInitializing = false;
                _initiallizeTaskCompletion?.TrySetException(ex);
                throw; // Re-throw exception
            }
        }

        /// <summary>
        /// Base class initialization processing.
        /// </summary>
        protected virtual void InitializeBase()
        {
            _cancellationTokenSource = new CancellationTokenSource();
        }

        /// <summary>
        /// Derived class synchronous initialization processing.
        /// </summary>
        protected virtual void InitializeCustom()
        {
            //Debug.LogWarning("_Initialize() called");
        }

        /// <summary>
        /// Derived class asynchronous initialization processing.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        protected virtual void InitializeAsyncCustom(CancellationToken cancellationToken)
        {
            //Debug.LogWarning("_Initialize() called");
        }

        /// <summary>
        /// Check if initialization is in progress.
        /// </summary>
        /// <returns>True if initializing</returns>
        public virtual bool IsInitializing()
        {
            return _isInitializing;
        }

        /// <summary>
        /// Check if already initialized.
        /// </summary>
        /// <returns>True if initialized</returns>
        public virtual bool IsInitialized()
        {
            return _hasInitDone;
        }

        /// <summary>
        /// Wait until initialization is complete.
        /// </summary>
        /// <returns>Awaitable task</returns>
        public virtual async Awaitable WaitForInitialized()
        {
            //Debug.LogWarning("WaitForInitialized() called");

            if (!_isInitializing || _initiallizeTaskCompletion == null)
                return;

            await _initiallizeTaskCompletion.Task;

            return;
        }

        /// <summary>
        /// Perform inference synchronously with specified texture. Returns TResult array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Inference results array</returns>
        public virtual TResult[] Infer(Texture texture)
        {
            //Debug.LogWarning("Infer() called");

            // Check for null texture
            if (texture == null)
                throw new ArgumentNullException(nameof(texture), "Texture cannot be null.");

            if (!_hasInitDone)
            {
                Debug.LogWarning("Initialize() must be called before Infer().");
                return null;
            }

            // Skip if disposing (Infer() will be disposed anyway)
            if (_isDisposing)
            {
                Debug.LogWarning("isDisposing is true. Infer() is not called.");
                return null;
            }

            if (_isInferring)
            {
                Debug.LogWarning("Infer(Texture) is already running. Request ignored.");
                return null; // Return null if already processing
            }

            _isInferring = true;
            _inferenceTaskCompletion = new TaskCompletionSource<bool>(); // Initialize task

            try
            {
                // Execute synchronous processing to get inference data
                TResult[] inferenceResults = InferCustom(texture);
                _inferenceTaskCompletion.TrySetResult(true); // Notify successful completion
                return inferenceResults;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Inference operation failed: {ex}");
                _inferenceTaskCompletion.TrySetException(ex); // Notify exception completion
                throw; // Re-throw exception
            }
            finally
            {
                _isInferring = false;
            }
        }

        /// <summary>
        /// Perform inference asynchronously with specified texture. Returns TResult array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Inference results array</returns>
        public virtual async Awaitable<TResult[]> InferAsync(Texture texture, CancellationToken cancellationToken = default)
        {
            //Debug.LogWarning("InferAsync() called");

            // Check for null texture
            if (texture == null)
                throw new ArgumentNullException(nameof(texture), "Texture cannot be null.");

            if (!_hasInitDone)
            {
                Debug.LogWarning("Initialize() must be called before DetectAsync().");
                return null;
            }

            // Skip if disposing (InferAsync() will be disposed anyway)
            if (_isDisposing)
            {
                Debug.LogWarning("isDisposing is true. InferAsync() is not called.");
                return null;
            }

            if (_isInferring)
            {
                Debug.LogWarning("InferAsync(Texture) is already running. Request ignored.");
                return null; // Return null if already processing
            }

            _isInferring = true;
            _inferenceTaskCompletion = new TaskCompletionSource<bool>(); // Initialize task

            using (var linkedCancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(_cancellationTokenSource.Token, cancellationToken))
            {
                try
                {
                    // Execute asynchronous processing to get inference data
                    TResult[] inferenceResults = await InferAsyncCustom(texture, linkedCancellationTokenSource.Token);
                    _inferenceTaskCompletion.TrySetResult(true); // Notify successful completion
                    return inferenceResults;
                }
                catch (OperationCanceledException)
                {
                    Debug.Log("Inference operation was canceled.");
                    _inferenceTaskCompletion.TrySetCanceled(); // Notify cancellation completion
                    return null; // Return null on cancellation
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Inference operation failed: {ex}");
                    _inferenceTaskCompletion.TrySetException(ex); // Notify exception completion
                    throw; // Re-throw exception
                }
                finally
                {
                    _isInferring = false;
                }
            }
        }



        /// <summary>
        /// Perform inference synchronously with specified Mat. Returns TResult array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <returns>Inference results array</returns>
        public virtual TResult[] Infer(Mat mat)
        {
            //Debug.LogWarning("Infer() called");

            // Check for null Mat
            if (mat == null)
                throw new ArgumentNullException(nameof(mat), "Mat cannot be null.");

            // Check if Mat is disposed
            if (mat != null) mat.ThrowIfDisposed();

            // Check Mat channel count (must be 4)
            if (mat.channels() != 4)
                throw new ArgumentException("Mat must have 4 channels.", nameof(mat));

            if (!_hasInitDone)
            {
                Debug.LogError("Initialize() must be called before Infer().");
                return null;
            }

            // Skip if disposing (Infer() will be disposed anyway)
            if (_isDisposing)
            {
                Debug.LogWarning("isDisposing is true. Infer() is not called.");
                return null;
            }

            if (_isInferring)
            {
                Debug.LogWarning("Infer(Mat) is already running. Request ignored.");
                return null; // Return null if already processing
            }

            _isInferring = true;
            _inferenceTaskCompletion = new TaskCompletionSource<bool>(); // Initialize task

            try
            {
                MatToRenderTexture(mat);

                // Execute synchronous processing to get inference data
                TResult[] inferenceResults = InferCustom(_renderTexture);
                _inferenceTaskCompletion.TrySetResult(true); // Notify successful completion
                return inferenceResults;
            }
            catch (Exception ex)
            {
                Debug.LogError($"Inference operation failed: {ex}");
                _inferenceTaskCompletion.TrySetException(ex); // Notify exception completion
                throw; // Re-throw exception
            }
            finally
            {
                _isInferring = false;
            }
        }

        /// <summary>
        /// Perform inference asynchronously with specified Mat. Returns TResult array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Inference results array</returns>
        public virtual async Awaitable<TResult[]> InferAsync(Mat mat, CancellationToken cancellationToken = default)
        {
            //Debug.LogWarning("InferAsync() called");

            // Check for null Mat
            if (mat == null)
                throw new ArgumentNullException(nameof(mat), "Mat cannot be null.");

            // Check if Mat is disposed
            if (mat != null) mat.ThrowIfDisposed();

            // Check Mat channel count (must be 4)
            if (mat.channels() != 4)
                throw new ArgumentException("Mat must have 4 channels.", nameof(mat));

            if (!_hasInitDone)
            {
                Debug.LogError("Initialize() must be called before DetectAsync().");
                return null;
            }

            // Skip if disposing (InferAsync() will be disposed anyway)
            if (_isDisposing)
            {
                Debug.LogWarning("isDisposing is true. InferAsync() is not called.");
                return null;
            }

            if (_isInferring)
            {
                Debug.LogWarning("InferAsync(Mat) is already running. Request ignored.");
                return null; // Return null if already processing
            }

            _isInferring = true;
            _inferenceTaskCompletion = new TaskCompletionSource<bool>(); // Initialize task

            using (var linkedCancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(_cancellationTokenSource.Token, cancellationToken))
            {
                try
                {
                    //await MatToRenderTextureAsync(mat, linkedCancellationTokenSource.Token);
                    MatToRenderTexture(mat);

                    linkedCancellationTokenSource.Token.ThrowIfCancellationRequested();

                    // Execute asynchronous processing to get inference data
                    TResult[] inferenceResults = await InferAsyncCustom(_renderTexture, linkedCancellationTokenSource.Token);
                    _inferenceTaskCompletion.TrySetResult(true); // Notify successful completion
                    return inferenceResults;
                }
                catch (OperationCanceledException)
                {
                    Debug.Log("Inference operation was canceled.");
                    _inferenceTaskCompletion.TrySetCanceled(); // Notify cancellation completion
                    return null; // Return null on cancellation
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Inference operation failed: {ex}");
                    _inferenceTaskCompletion.TrySetException(ex); // Notify exception completion
                    throw; // Re-throw exception
                }
                finally
                {
                    _isInferring = false;
                }
            }
        }

        /// <summary>
        /// Implementation of synchronous inference processing for derived classes.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Inference results array</returns>
        protected abstract TResult[] InferCustom(Texture texture);

        /// <summary>
        /// Implementation of asynchronous inference processing for derived classes.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Inference results array</returns>
        protected abstract Awaitable<TResult[]> InferAsyncCustom(Texture texture, CancellationToken cancellationToken);

        /// <summary>
        /// Check if inference is in progress.
        /// </summary>
        /// <returns>True if inferring</returns>
        public virtual bool IsInferring()
        {
            return _isInferring;
        }

        /// <summary>
        /// Check if inference has finished.
        /// </summary>
        /// <returns>True if inference is finished</returns>
        public virtual bool IsInferred()
        {
            return !_isInferring;
        }

        /// <summary>
        /// Wait until inference is complete.
        /// </summary>
        /// <returns>Awaitable task</returns>
        public virtual async Awaitable WaitForInferred()
        {
            //Debug.LogWarning("WaitForInferred() called");

            if (!_isInferring || _inferenceTaskCompletion == null)
                return;
            await _inferenceTaskCompletion.Task;
            return;
        }

        /// <summary>
        /// Check if inference can be started.
        /// </summary>
        /// <returns>True if inference can be started</returns>
        public virtual bool CanStartInference()
        {
            // Cannot start if not initialized
            if (!_hasInitDone)
            {
                return false;
            }

            // Cannot start if disposing
            if (_isDisposing)
            {
                return false;
            }

            // Cannot start if already inferring
            if (_isInferring)
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Check if dispose is in progress.
        /// </summary>
        /// <returns>True if disposing</returns>
        public virtual bool IsDisposing()
        {
            return _isDisposing;
        }

        /// <summary>
        /// Check if dispose has completed.
        /// </summary>
        /// <returns>True if disposed</returns>
        public virtual bool IsDisposed()
        {
            return !_isDisposing && !_hasInitDone;
        }

        /// <summary>
        /// Perform dispose processing synchronously.
        /// </summary>
        public virtual void Dispose()
        {
            //Debug.LogWarning("Dispose() called");

            // Skip if already disposing to avoid double processing
            if (_isDisposing)
            {
                Debug.LogWarning("Dispose() is already running.");
                return;
            }

            // Skip if initializing
            if (_isInitializing)
            {
                Debug.LogWarning("Initialize() is running. Dispose() is not called.");
                return;
            }

            // Skip if inferring
            if (_isInferring)
            {
                Debug.LogWarning("Infer() is running. Dispose() is not called.");
                return;
            }

            _isDisposing = true; // Set disposing flag for `Dispose()`
            _disposeTaskCompletion = new TaskCompletionSource<bool>(); // Create new task

            try
            {
                DisposeCustom();
                DisposeBase();

                _hasInitDone = false;

                _isDisposing = false; // Notify completion of `Dispose()`
                _disposeTaskCompletion?.TrySetResult(true); // Notify completion to `Initialize()`

                //Debug.Log("Dispose() finished");
            }
            catch (Exception ex)
            {
                // Cleanup state on exception
                _isDisposing = false;
                _disposeTaskCompletion?.TrySetException(ex);
                throw; // Re-throw exception
            }
        }

        /// <summary>
        /// Perform dispose processing asynchronously.
        /// </summary>
        /// <returns>Awaitable task</returns>
        public virtual async Awaitable DisposeAsync()
        {
            //Debug.LogWarning("DisposeAsync() called");

            if (_isDisposing) return; // Skip if already disposing to avoid double processing

            _isDisposing = true; // Set disposing flag for `DisposeAsync()`
            _disposeTaskCompletion = new TaskCompletionSource<bool>(); // Create new task

            // Wait for InitializeAsync() to complete if not initialized and initializing (not called from within InitializeAsync)
            if (!_hasInitDone && _isInitializing)
            {
                //Debug.LogWarning("DisposeAsync() is waiting for InitializeAsync() to complete.");
                await _initiallizeTaskCompletion.Task; // Wait for `InitializeAsync()` completion
            }

            if (_isInferring)
            {
                _cancellationTokenSource?.Cancel();
                //Debug.Log("DisposeAsync: Cancelling InferAsync and waiting...");

                try
                {
                    // Wait for inference to complete (wait for completion even if canceled)
                    await _inferenceTaskCompletion.Task;
                }
                catch (OperationCanceledException)
                {
                    // Cancellation is expected behavior, so no log output
                    Debug.Log("DisposeAsync: Inference was canceled as expected.");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"DisposeAsync: Exception occurred while waiting for inference to complete - {ex}");
                }
            }

            DisposeAsyncCustom();
            DisposeBase();

            _hasInitDone = false;

            _isDisposing = false; // Notify completion of `DisposeAsync()`
            _disposeTaskCompletion?.TrySetResult(true); // Notify completion to `InitializeAsync()`

            //Debug.Log("DisposeAsync() finished");
        }

        /// <summary>
        /// Base class dispose processing.
        /// </summary>
        protected void DisposeBase()
        {
            // Release RenderTexture and GraphicsBuffer
            if (_renderTexture != null && _renderTexture.IsCreated())
            {
                _renderTexture.Release();
                _renderTexture = null;
            }
            _graphicsBuffer?.Dispose();
            _graphicsBuffer = null;

            _cancellationTokenSource?.Dispose();
            _cancellationTokenSource = null;

            // Dispose event
            OnInitialize = null;
        }

        /// <summary>
        /// Derived class synchronous dispose processing.
        /// </summary>
        protected virtual void DisposeCustom()
        {
            //Debug.LogWarning("_Dispose() called");
        }

        /// <summary>
        /// Derived class asynchronous dispose processing.
        /// </summary>
        protected virtual void DisposeAsyncCustom()
        {
            //Debug.LogWarning("_Dispose() called");
        }

        /// <summary>
        /// Wait until dispose is complete.
        /// </summary>
        /// <returns>Awaitable task</returns>
        public virtual async Awaitable WaitForDisposed()
        {
            //Debug.LogWarning("WaitForDisposed() called");

            if (!_isDisposing || _disposeTaskCompletion == null)
                return;
            await _disposeTaskCompletion.Task;
            return;
        }

        #endregion

        #region Protected Methods

        #endregion

        #region Private Methods

        /// <summary>
        /// Create RenderTexture and GraphicsBuffer with the same size as the specified Mat.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        private void InitializeRenderTextureAndGraphicsBuffer(Mat mat)
        {
            if (mat == null)
                throw new ArgumentNullException(nameof(mat), "Mat cannot be null.");

            if (_renderTexture == null || _renderTexture.width != mat.cols() || _renderTexture.height != mat.rows())
            {
                if (_renderTexture != null)
                {
                    _renderTexture.Release();
                    _renderTexture = null;
                }

                _renderTexture = new RenderTexture(mat.width(), mat.height(), 0);
                _renderTexture.enableRandomWrite = true;
                _renderTexture.Create();

                if (_graphicsBuffer != null)
                {
                    _graphicsBuffer.Dispose();
                    _graphicsBuffer = null;
                }
                _graphicsBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, (int)mat.total(), (int)mat.elemSize());
            }
        }

        /// <summary>
        /// Create RenderTexture and GraphicsBuffer with the same size as the specified Mat, then copy Mat data to RenderTexture.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        private void MatToRenderTexture(Mat mat)
        {
            InitializeRenderTextureAndGraphicsBuffer(mat);

            OpenCVMatUtils.MatToRenderTexture(mat, _renderTexture, _graphicsBuffer);
        }

        /// <summary>
        /// Create RenderTexture and GraphicsBuffer with the same size as the specified Mat, then copy Mat data to RenderTexture asynchronously.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Awaitable task</returns>
        private async Awaitable MatToRenderTextureAsync(Mat mat, CancellationToken cancellationToken = default)
        {
            // Check for cancellation
            cancellationToken.ThrowIfCancellationRequested();

            InitializeRenderTextureAndGraphicsBuffer(mat);

            // Check for cancellation again (after resource creation)
            cancellationToken.ThrowIfCancellationRequested();

            await OpenCVMatUtils.MatToRenderTextureAsync(mat, _renderTexture, _graphicsBuffer);

            //Debug.Log("MatToRenderTextureAsync finished");
        }

        #endregion
    }
}
