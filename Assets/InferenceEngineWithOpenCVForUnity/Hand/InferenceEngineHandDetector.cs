using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using Unity.InferenceEngine;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Events;

namespace InferenceEngineWithOpenCVForUnity.Hand
{
    /// <summary>
    /// Hand detector using UnityInferenceEngine for hand detection and landmark estimation.
    /// </summary>
    public class InferenceEngineHandDetector : InferenceEngineManager<InferenceEngineHandDetector.EstimationData>
    {
        #region Fields

        // 1. Constants
        private const int K_NUM_ANCHORS = 2016;
        private const int K_NUM_KEYPOINTS = 21;
        private const int DETECTOR_INPUT_SIZE = 192;
        private const int LANDMARKER_INPUT_SIZE = 224;

        // 2. Static fields
        protected static readonly Scalar SCALAR_WHITE = new Scalar(255, 255, 255, 255);
        protected static readonly Scalar SCALAR_RED = new Scalar(0, 0, 255, 255);
        protected static readonly Scalar SCALAR_GREEN = new Scalar(0, 255, 0, 255);
        protected static readonly Scalar SCALAR_BLUE = new Scalar(255, 0, 0, 255);

        // 3. Serialized instance fields
        [SerializeField]
        private ModelAsset _handDetector;
        [SerializeField]
        private ModelAsset _handLandmarker;
        [SerializeField]
        private TextAsset _anchorsCSV;
        [SerializeField]
        private float _scoreThreshold = 0.5f;
        [SerializeField]
        private float _iouThreshold = 0.5f;
        [SerializeField]
        private bool _useBestHandOnly = false;

        // 4. Instance fields
        private float[,] _anchors;
        private Worker _handDetectorWorker;
        private Worker _handLandmarkerWorker;
        private Tensor<float> _detectorInput;
        private Tensor<float> _landmarkerInput;
        private float _textureWidth;
        private float _textureHeight;

        // 5. Protected fields
        public UnityEvent<EstimationData[]> OnDetectFinished = new UnityEvent<EstimationData[]>(); // Event notification for multiple DetectionData

        // Properties
        public ModelAsset HandDetector
        {
            get => _handDetector;
            set => _handDetector = value;
        }

        public ModelAsset HandLandmarker
        {
            get => _handLandmarker;
            set => _handLandmarker = value;
        }

        public TextAsset AnchorsCSV
        {
            get => _anchorsCSV;
            set => _anchorsCSV = value;
        }

        public float ScoreThreshold
        {
            get => _scoreThreshold;
            set => _scoreThreshold = value;
        }

        public float IouThreshold
        {
            get => _iouThreshold;
            set => _iouThreshold = value;
        }

        public bool UseBestHandOnly
        {
            get => _useBestHandOnly;
            set => _useBestHandOnly = value;
        }

        #endregion

        #region Enums

        /// <summary>
        /// Hand key point enumeration.
        /// </summary>
        public enum KeyPoint
        {
            Wrist,
            Thumb1, Thumb2, Thumb3, Thumb4,
            Index1, Index2, Index3, Index4,
            Middle1, Middle2, Middle3, Middle4,
            Ring1, Ring2, Ring3, Ring4,
            Pinky1, Pinky2, Pinky3, Pinky4
        }

        #endregion

        #region Data Structures

        /// <summary>
        /// Hand detection and landmark estimation data structure.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public readonly struct EstimationData
        {
            public readonly float X1;
            public readonly float Y1;
            public readonly float X2;
            public readonly float Y2;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = LANDMARK_ELEMENT_COUNT)]
            private readonly float[] _rawLandmarksScreen;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = LANDMARK_ELEMENT_COUNT)]
            private readonly float[] _rawLandmarksWorld;

            public readonly float Handedness;
            public readonly float Confidence;

            public const int LANDMARK_VEC3F_COUNT = 21;
            public const int LANDMARK_ELEMENT_COUNT = 3 * LANDMARK_VEC3F_COUNT;
            public const int ELEMENT_COUNT = 4 + LANDMARK_ELEMENT_COUNT + LANDMARK_ELEMENT_COUNT + 2;
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Initializes a new instance of the EstimationData struct.
            /// </summary>
            /// <param name="x1">Bounding box X1 coordinate</param>
            /// <param name="y1">Bounding box Y1 coordinate</param>
            /// <param name="x2">Bounding box X2 coordinate</param>
            /// <param name="y2">Bounding box Y2 coordinate</param>
            /// <param name="landmarksScreen">Screen landmarks</param>
            /// <param name="landmarksWorld">World landmarks</param>
            /// <param name="handedness">Handedness value</param>
            /// <param name="confidence">Confidence score</param>
            public EstimationData(float x1, float y1, float x2, float y2,
                                 Vec3f[] landmarksScreen, Vec3f[] landmarksWorld,
                                 float handedness, float confidence)
            {
                if (landmarksScreen == null || landmarksScreen.Length != LANDMARK_VEC3F_COUNT)
                    throw new ArgumentException("landmarksScreen must be a Vec3f[" + LANDMARK_VEC3F_COUNT + "]");
                if (landmarksWorld == null || landmarksWorld.Length != LANDMARK_VEC3F_COUNT)
                    throw new ArgumentException("landmarksWorld must be a Vec3f[" + LANDMARK_VEC3F_COUNT + "]");

                X1 = x1;
                Y1 = y1;
                X2 = x2;
                Y2 = y2;
                _rawLandmarksScreen = new float[LANDMARK_ELEMENT_COUNT];
                for (int i = 0; i < landmarksScreen.Length; i++)
                {
                    int offset = i * 3;
                    ref readonly var landmark = ref landmarksScreen[i];
                    _rawLandmarksScreen[offset + 0] = landmark.Item1;
                    _rawLandmarksScreen[offset + 1] = landmark.Item2;
                    _rawLandmarksScreen[offset + 2] = landmark.Item3;
                }
                _rawLandmarksWorld = new float[LANDMARK_ELEMENT_COUNT];
                for (int i = 0; i < landmarksWorld.Length; i++)
                {
                    int offset = i * 3;
                    ref readonly var landmark = ref landmarksWorld[i];
                    _rawLandmarksWorld[offset + 0] = landmark.Item1;
                    _rawLandmarksWorld[offset + 1] = landmark.Item2;
                    _rawLandmarksWorld[offset + 2] = landmark.Item3;
                }
                Handedness = handedness;
                Confidence = confidence;
            }

#if NET_STANDARD_2_1

            public readonly ReadOnlySpan<Vec3f> GetLandmarksScreen()
            {
                return MemoryMarshal.Cast<float, Vec3f>(_rawLandmarksScreen.AsSpan());
            }

            public readonly ReadOnlySpan<Vec3f> GetLandmarksWorld()
            {
                return MemoryMarshal.Cast<float, Vec3f>(_rawLandmarksWorld.AsSpan());
            }

#endif

            public readonly Vec3f[] GetLandmarksScreenArray()
            {
                Vec3f[] landmarks = new Vec3f[LANDMARK_VEC3F_COUNT];
                for (int i = 0; i < landmarks.Length; i++)
                {
                    int offset = i * 3;
                    landmarks[i] = new Vec3f(_rawLandmarksScreen[offset + 0],
                                             _rawLandmarksScreen[offset + 1],
                                             _rawLandmarksScreen[offset + 2]);
                }
                return landmarks;
            }

            public readonly Vec3f[] GetLandmarksWorldArray()
            {
                Vec3f[] landmarks = new Vec3f[LANDMARK_VEC3F_COUNT];
                for (int i = 0; i < landmarks.Length; i++)
                {
                    int offset = i * 3;
                    landmarks[i] = new Vec3f(_rawLandmarksWorld[offset + 0],
                                             _rawLandmarksWorld[offset + 1],
                                             _rawLandmarksWorld[offset + 2]);
                }
                return landmarks;
            }

            /// <summary>
            /// Returns a string representation of the EstimationData.
            /// </summary>
            /// <returns>String representation</returns>
            public readonly override string ToString()
            {
                StringBuilder sb = new StringBuilder(1536);

                sb.Append("EstimationData(");
                sb.AppendFormat("X1:{0} Y1:{1} X2:{2} Y2:{3} ", X1, Y1, X2, Y2);

                sb.Append("LandmarksScreen:");
#if NET_STANDARD_2_1
                ReadOnlySpan<Vec3f> landmarksScreen = GetLandmarksScreen();
#else
                Vec3f[] landmarksScreen = GetLandmarksScreenArray();
#endif
                for (int i = 0; i < landmarksScreen.Length; i++)
                {
                    ref readonly var p = ref landmarksScreen[i];
                    sb.Append(p.ToString());
                }
                sb.Append(" ");

                sb.Append("LandmarksWorld:");
#if NET_STANDARD_2_1
                ReadOnlySpan<Vec3f> landmarksWorld = GetLandmarksWorld();
#else
                Vec3f[] landmarksWorld = GetLandmarksWorldArray();
#endif
                for (int i = 0; i < landmarksWorld.Length; i++)
                {
                    ref readonly var p = ref landmarksWorld[i];
                    sb.Append(p.ToString());
                }
                sb.Append(" ");

                sb.AppendFormat("Handedness:{0},({1}) ", Handedness, Handedness <= 0.5f ? "Left" : "Right");
                sb.AppendFormat("Confidence:{0}", Confidence);
                sb.Append(")");
                return sb.ToString();
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the hand detector with specified parameters.
        /// </summary>
        /// <param name="handDetector">Hand detector model asset</param>
        /// <param name="handLandmarker">Hand landmarker model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestHandOnly">Whether to use only the best hand</param>
        public void Initialize(ModelAsset handDetector, ModelAsset handLandmarker, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestHandOnly)
        {
            OnInitialize += () => ApplyInitializationData(handDetector, handLandmarker, anchorsCSV, iouThreshold, scoreThreshold, useBestHandOnly);
            Initialize();
        }

        /// <summary>
        /// Initialize the hand detector asynchronously with specified parameters.
        /// </summary>
        /// <param name="handDetector">Hand detector model asset</param>
        /// <param name="handLandmarker">Hand landmarker model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestHandOnly">Whether to use only the best hand</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Awaitable task</returns>
        public async Awaitable InitializeAsync(ModelAsset handDetector, ModelAsset handLandmarker, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestHandOnly, CancellationToken cancellationToken = default)
        {
            OnInitialize += () => ApplyInitializationData(handDetector, handLandmarker, anchorsCSV, iouThreshold, scoreThreshold, useBestHandOnly);
            await InitializeAsync(cancellationToken);
        }

        /// <summary>
        /// Perform hand detection synchronously with specified texture. Returns EstimationData array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Hand detection results array</returns>
        public override InferenceEngineHandDetector.EstimationData[] Infer(Texture texture)
        {
            // Get results
            InferenceEngineHandDetector.EstimationData[] result = base.Infer(texture);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform hand detection synchronously with specified Mat. Returns EstimationData array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <returns>Hand detection results array</returns>
        public override InferenceEngineHandDetector.EstimationData[] Infer(Mat mat)
        {
            // Get results
            InferenceEngineHandDetector.EstimationData[] result = base.Infer(mat);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform hand detection asynchronously with specified texture. Returns EstimationData array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Hand detection results array</returns>
        public override async Awaitable<InferenceEngineHandDetector.EstimationData[]> InferAsync(Texture texture, CancellationToken cancellationToken = default)
        {
            // Get results
            InferenceEngineHandDetector.EstimationData[] result = await base.InferAsync(texture, cancellationToken);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform hand detection asynchronously with specified Mat. Returns EstimationData array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Hand detection results array</returns>
        public override async Awaitable<InferenceEngineHandDetector.EstimationData[]> InferAsync(Mat mat, CancellationToken cancellationToken = default)
        {
            // Get results
            InferenceEngineHandDetector.EstimationData[] result = await base.InferAsync(mat, cancellationToken);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Custom initialization for hand detector.
        /// </summary>
        protected override void InitializeCustom()
        {
            //Debug.LogWarning("_Initialize() called");

            LoadAnchors();
            var handDetectorModel = CreateHandDetectorModel();
            var handLandmarkerModel = CreateHandLandmarkerModel();
            InitializeWorkersAndTensors(handDetectorModel, handLandmarkerModel);
        }

        /// <summary>
        /// Custom asynchronous initialization for hand detector.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        protected override void InitializeAsyncCustom(CancellationToken cancellationToken)
        {
            //Debug.LogWarning("_Initialize() called");

            LoadAnchors(cancellationToken);
            var handDetectorModel = CreateHandDetectorModel(cancellationToken);
            var handLandmarkerModel = CreateHandLandmarkerModel(cancellationToken);
            InitializeWorkersAndTensors(handDetectorModel, handLandmarkerModel, cancellationToken);
        }

        /// <summary>
        /// Custom hand detection implementation for texture input.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Hand detection results array</returns>
#if !UNITY_WEBGL
        protected override InferenceEngineHandDetector.EstimationData[] InferCustom(Texture texture)
        {
            var M = PrepareTextureAndScheduleInference(texture);

            // Get tensors synchronously
            var (outputIdx, outputScore, outputBox) = ReadHandDetectorTensors();

            using (outputIdx)
            using (outputScore)
            using (outputBox)
            {
                return ProcessHandDetectionResults(outputIdx, outputScore, outputBox, M, texture);
            }
        }
#else
        protected override InferenceEngineHandDetector.EstimationData[] InferCustom(Texture texture)
        {
            Debug.LogWarning("Infer is not supported on WebGL platform due to ReadbackAndClone limitations.");
            return null;
        }
#endif

        /// <summary>
        /// Custom asynchronous hand detection implementation for texture input.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Hand detection results array</returns>
        protected override async Awaitable<InferenceEngineHandDetector.EstimationData[]> InferAsyncCustom(Texture texture, CancellationToken cancellationToken)
        {
            var M = PrepareTextureAndScheduleInference(texture);

            var (outputIdx, outputScore, outputBox) = await ReadHandDetectorTensorsAsync(cancellationToken);

            using (outputIdx)
            using (outputScore)
            using (outputBox)
            {
                return await ProcessHandDetectionResultsAsync(outputIdx, outputScore, outputBox, M, texture, cancellationToken);
            }
        }

        /// <summary>
        /// Custom dispose processing for hand detector.
        /// </summary>
        protected override void DisposeCustom()
        {
            DisposeResources();
        }

        /// <summary>
        /// Custom asynchronous dispose processing for hand detector.
        /// </summary>
        protected override void DisposeAsyncCustom()
        {
            DisposeResources();
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Loads anchors from CSV text data
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        private void LoadAnchors(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _anchors = BlazeUtils.LoadAnchors(_anchorsCSV.text, K_NUM_ANCHORS);
        }

        /// <summary>
        /// Creates and configures the hand detector model with post-processing
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>The compiled hand detector model</returns>
        private Model CreateHandDetectorModel(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var handDetectorModel = ModelLoader.Load(_handDetector);

            var graph = new FunctionalGraph();
            var input = graph.AddInput(handDetectorModel, 0);
            var outputs = Functional.Forward(handDetectorModel, input);
            var boxes = outputs[0]; // (1, 2016, 18)
            var scores = outputs[1]; // (1, 2016, 1)

            if (_useBestHandOnly)
            {
                // post process the model to filter scores + argmax select the best hand
                var idx_scores_boxes = BlazeUtils.ArgMaxFiltering(boxes, scores);
                return graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);
            }
            else
            {
                // post process the model to filter scores + nms select the best hands
                var anchorsData = new float[K_NUM_ANCHORS * 4];
                Buffer.BlockCopy(_anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
                var anchors = Functional.Constant(new TensorShape(K_NUM_ANCHORS, 4), anchorsData);
                var idx_scores_boxes = BlazeUtils.NMSFiltering(boxes, scores, anchors, DETECTOR_INPUT_SIZE, _iouThreshold, _scoreThreshold);
                return graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);
            }
        }

        /// <summary>
        /// Creates and configures the hand landmarker model
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>The hand landmarker model</returns>
        private Model CreateHandLandmarkerModel(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ModelLoader.Load(_handLandmarker);
        }

        /// <summary>
        /// Initializes the workers and input tensors
        /// </summary>
        /// <param name="handDetectorModel">The compiled hand detector model</param>
        /// <param name="handLandmarkerModel">The hand landmarker model</param>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        private void InitializeWorkersAndTensors(Model handDetectorModel, Model handLandmarkerModel, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _handDetectorWorker = new Worker(handDetectorModel, BackendType);
            _handLandmarkerWorker = new Worker(handLandmarkerModel, BackendType);
            _detectorInput = new Tensor<float>(new TensorShape(1, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3));
            _landmarkerInput = new Tensor<float>(new TensorShape(1, LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE, 3));
        }

        /// <summary>
        /// Prepares texture for inference by calculating affine transformation, sampling, and scheduling inference
        /// </summary>
        /// <param name="texture">Input texture to process</param>
        /// <returns>Affine transformation matrix for coordinate conversion</returns>
        private float2x3 PrepareTextureAndScheduleInference(Texture texture)
        {
            _textureWidth = texture.width;
            _textureHeight = texture.height;

            var size = Mathf.Max(texture.width, texture.height);

            // The affine transformation matrix to go from tensor coordinates to image coordinates
            var scale = size / (float)DETECTOR_INPUT_SIZE;
            var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (new float2(texture.width, texture.height) + new float2(-size, size))), BlazeUtils.ScaleMatrix(new float2(scale, -scale)));
            BlazeUtils.SampleImageAffine(texture, _detectorInput, M);

            // Schedule the inference
            _handDetectorWorker.Schedule(_detectorInput);

            return M;
        }

        /// <summary>
        /// Reads hand detector tensors synchronously
        /// </summary>
        /// <returns>Tuple containing the read hand detector tensors</returns>
        private (Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox) ReadHandDetectorTensors()
        {
            var outputIdx = (_handDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndClone();
            var outputScore = (_handDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndClone();
            var outputBox = (_handDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndClone();

            return (outputIdx, outputScore, outputBox);
        }

        /// <summary>
        /// Reads hand detector tensors asynchronously with cancellation support
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Tuple containing the read hand detector tensors</returns>
        private async Awaitable<(Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox)> ReadHandDetectorTensorsAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var outputIdxAwaitable = (_handDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
            var outputScoreAwaitable = (_handDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
            var outputBoxAwaitable = (_handDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

            Tensor<int> outputIdx = null;
            Tensor<float> outputScore = null;
            Tensor<float> outputBox = null;

            try
            {
                outputIdx = await outputIdxAwaitable;
                outputScore = await outputScoreAwaitable;
                outputBox = await outputBoxAwaitable;
                cancellationToken.ThrowIfCancellationRequested();

                return (outputIdx, outputScore, outputBox);
            }
            catch (OperationCanceledException)
            {
                Debug.LogWarning("Hand detector tensor reading was cancelled in ReadHandDetectorTensorsAsync");
                // Ensure tensors are disposed if cancellation occurred
                outputIdx?.Dispose();
                outputScore?.Dispose();
                outputBox?.Dispose();

                throw;
            }
        }

        /// <summary>
        /// Reads hand landmarker tensors synchronously
        /// </summary>
        /// <returns>Tuple containing the read hand landmarker tensors</returns>
        private (Tensor<float> landmarks_screen, Tensor<float> confidence, Tensor<float> handedness, Tensor<float> landmarks_world) ReadHandLandmarkerTensors()
        {
            var landmarks_screen = (_handLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndClone();
            var confidence = (_handLandmarkerWorker.PeekOutput("Identity_1") as Tensor<float>).ReadbackAndClone();
            var handedness = (_handLandmarkerWorker.PeekOutput("Identity_2") as Tensor<float>).ReadbackAndClone();
            var landmarks_world = (_handLandmarkerWorker.PeekOutput("Identity_3") as Tensor<float>).ReadbackAndClone();

            return (landmarks_screen, confidence, handedness, landmarks_world);
        }

        /// <summary>
        /// Reads hand landmarker tensors asynchronously with cancellation support
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Tuple containing the read hand landmarker tensors</returns>
        private async Awaitable<(Tensor<float> landmarks_screen, Tensor<float> confidence, Tensor<float> handedness, Tensor<float> landmarks_world)> ReadHandLandmarkerTensorsAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var landmarks_screenAwaitable = (_handLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
            var confidenceAwaitable = (_handLandmarkerWorker.PeekOutput("Identity_1") as Tensor<float>).ReadbackAndCloneAsync();
            var handednessAwaitable = (_handLandmarkerWorker.PeekOutput("Identity_2") as Tensor<float>).ReadbackAndCloneAsync();
            var landmarks_worldAwaitable = (_handLandmarkerWorker.PeekOutput("Identity_3") as Tensor<float>).ReadbackAndCloneAsync();

            Tensor<float> landmarks_screen = null;
            Tensor<float> confidence = null;
            Tensor<float> handedness = null;
            Tensor<float> landmarks_world = null;

            try
            {
                landmarks_screen = await landmarks_screenAwaitable;
                confidence = await confidenceAwaitable;
                handedness = await handednessAwaitable;
                landmarks_world = await landmarks_worldAwaitable;
                cancellationToken.ThrowIfCancellationRequested();

                return (landmarks_screen, confidence, handedness, landmarks_world);
            }
            catch (OperationCanceledException)
            {
                Debug.LogWarning("Hand landmarker tensor reading was cancelled in ReadHandLandmarkerTensorsAsync");
                // Ensure tensors are disposed if cancellation occurred
                landmarks_screen?.Dispose();
                confidence?.Dispose();
                handedness?.Dispose();
                landmarks_world?.Dispose();

                throw;
            }
        }

        /// <summary>
        /// Process hand landmarker preparation for a single detection
        /// </summary>
        /// <param name="idx">Anchor index</param>
        /// <param name="outputBox">Output box tensor</param>
        /// <param name="M">Transformation matrix</param>
        /// <param name="texture">Input texture</param>
        /// <returns>Tuple containing M2 matrix and rotation value</returns>
        private (float2x3 M2, float rotation) ProcessHandLandmarkerPreparation(int idx, Tensor<float> outputBox, float2x3 M, Texture texture)
        {
            // SIMD最適化：anchorPositionの計算
            var anchorX = _anchors[idx, 0];
            var anchorY = _anchors[idx, 1];
            var anchorPosition = DETECTOR_INPUT_SIZE * new float2(anchorX, anchorY);

            // SIMD最適化：box座標をfloat4で一括処理
            var boxCoordinates = new float4(
                outputBox[0, 0, 0],  // boxCentre X
                outputBox[0, 0, 1],  // boxCentre Y
                outputBox[0, 0, 2],  // boxSize X
                outputBox[0, 0, 3]   // boxSize Y
            );

            var boxCentre_TensorSpace = anchorPosition + new float2(boxCoordinates.x, boxCoordinates.y);
            var boxSize_TensorSpace = math.max(boxCoordinates.z, boxCoordinates.w);

            // SIMD最適化：キーポイント座標をfloat2で処理
            var keypoint0 = new float2(
                outputBox[0, 0, 4 + 2 * 0 + 0],  // kp0 X
                outputBox[0, 0, 4 + 2 * 0 + 1]   // kp0 Y
            );
            var keypoint2 = new float2(
                outputBox[0, 0, 4 + 2 * 2 + 0],  // kp2 X
                outputBox[0, 0, 4 + 2 * 2 + 1]   // kp2 Y
            );

            var kp0_TensorSpace = anchorPosition + keypoint0;
            var kp2_TensorSpace = anchorPosition + keypoint2;
            var delta_TensorSpace = kp2_TensorSpace - kp0_TensorSpace;
            var up_TensorSpace = delta_TensorSpace / math.length(delta_TensorSpace);
            var theta = math.atan2(delta_TensorSpace.y, delta_TensorSpace.x);
            var rotation = 0.5f * Mathf.PI - theta;
            boxCentre_TensorSpace += 0.5f * boxSize_TensorSpace * up_TensorSpace;
            boxSize_TensorSpace *= 2.6f;

            var origin2 = new float2(0.5f * LANDMARKER_INPUT_SIZE, 0.5f * LANDMARKER_INPUT_SIZE);
            var scale2 = boxSize_TensorSpace / LANDMARKER_INPUT_SIZE;
            var M2 = BlazeUtils.mul(M, BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.TranslationMatrix(boxCentre_TensorSpace), BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))), BlazeUtils.RotationMatrix(rotation)), BlazeUtils.TranslationMatrix(-origin2)));
            BlazeUtils.SampleImageAffine(texture, _landmarkerInput, M2);

            _handLandmarkerWorker.Schedule(_landmarkerInput);

            return (M2, rotation);
        }

        /// <summary>
        /// Process hand landmarks and calculate bounding box
        /// </summary>
        /// <param name="landmarks_screen">Screen landmarks tensor</param>
        /// <param name="landmarks_world">World landmarks tensor</param>
        /// <param name="M2">Transformation matrix for screen coordinates</param>
        /// <param name="rotation">Rotation value</param>
        /// <returns>Tuple containing screen landmarks, world landmarks, and bounding box</returns>
        private (Vec3f[] landmarks_screen_vec3f, Vec3f[] landmarks_world_vec3f, OpenCVForUnity.CoreModule.Rect bbox) ProcessHandLandmarksAndBoundingBox(Tensor<float> landmarks_screen, Tensor<float> landmarks_world, float2x3 M2, float rotation)
        {
            var M3 = BlazeUtils.RotationMatrix(rotation);

            // SIMD最適化：キーポイント処理のベクトル化
            Vec3f[] landmarks_screen_vec3f = new Vec3f[K_NUM_KEYPOINTS];
            Vec3f[] landmarks_world_vec3f = new Vec3f[K_NUM_KEYPOINTS];

            // SIMD最適化：21個のキーポイントをfloat3で処理
            for (var i = 0; i < K_NUM_KEYPOINTS; i++)
            {
                // SIMD最適化：3つのfloat値を同時に処理（float3使用）
                var landmark_screen_data = new float3(
                    landmarks_screen[3 * i + 0],
                    landmarks_screen[3 * i + 1],
                    landmarks_screen[3 * i + 2]
                );
                var landmark_world_data = new float3(
                    landmarks_world[3 * i + 0],
                    landmarks_world[3 * i + 1],
                    landmarks_world[3 * i + 2]
                );

                float3 landmark_screen = BlazeUtils.mul(M2, landmark_screen_data);
                landmarks_screen_vec3f[i] = new Vec3f(landmark_screen.x, landmark_screen.y, landmark_screen.z);
                // OpenCVの座標系に合わせる
                landmarks_screen_vec3f[i].Item2 = _textureHeight - landmarks_screen_vec3f[i].Item2;

                float3 landmark_world = BlazeUtils.mul(M3, landmark_world_data);
                landmarks_world_vec3f[i] = new Vec3f(landmark_world.x, landmark_world.y, landmark_world.z);
            }

            // SIMD最適化：Vec2f配列の作成を最適化
            Vec2f[] landmarks_screen_vec2f = new Vec2f[landmarks_screen_vec3f.Length];
            for (int i = 0; i < landmarks_screen_vec2f.Length; i++)
            {
                landmarks_screen_vec2f[i] = new Vec2f(landmarks_screen_vec3f[i].Item1, landmarks_screen_vec3f[i].Item2);
            }
            MatOfPoint points = new MatOfPoint(landmarks_screen_vec2f);
            OpenCVForUnity.CoreModule.Rect bbox = Imgproc.boundingRect(points);
            points.Dispose();

            // Enlarge bounding box to 2x size
            bbox = EnlargeBoundingBox(bbox, 1.5);

            // Crop bounding box to intersection with texture boundaries
            bbox = ClampBoundingBoxToImage(bbox);

            return (landmarks_screen_vec3f, landmarks_world_vec3f, bbox);
        }

        /// <summary>
        /// Enlarges bounding box by specified factor while keeping center position
        /// </summary>
        /// <param name="bbox">Original bounding box</param>
        /// <param name="factor">Enlargement factor</param>
        /// <returns>Enlarged bounding box</returns>
        private OpenCVForUnity.CoreModule.Rect EnlargeBoundingBox(OpenCVForUnity.CoreModule.Rect bbox, double factor)
        {
            // Calculate center point
            Point center = new Point(bbox.x + bbox.width / 2.0, bbox.y + bbox.height / 2.0);

            // Calculate new dimensions
            int newWidth = (int)(bbox.width * factor);
            int newHeight = (int)(bbox.height * factor);

            // Calculate new top-left position to keep center
            int newX = (int)(center.x - newWidth / 2.0);
            int newY = (int)(center.y - newHeight / 2.0);

            return new OpenCVForUnity.CoreModule.Rect(newX, newY, newWidth, newHeight);
        }

        /// <summary>
        /// Crops bounding box to intersection with texture boundaries (0, 0, _textureWidth, _textureHeight)
        /// </summary>
        /// <param name="bbox">Bounding box to crop</param>
        /// <returns>Cropped bounding box (intersection with texture)</returns>
        private OpenCVForUnity.CoreModule.Rect ClampBoundingBoxToImage(OpenCVForUnity.CoreModule.Rect bbox)
        {
            // Create texture rectangle
            OpenCVForUnity.CoreModule.Rect textureRect = new OpenCVForUnity.CoreModule.Rect(0, 0, (int)_textureWidth, (int)_textureHeight);

            // Get intersection of bbox and texture rectangle
            OpenCVForUnity.CoreModule.Rect intersection = bbox.intersect(textureRect);

            // If no intersection, return a minimal rectangle at (0,0)
            if (intersection.width <= 0 || intersection.height <= 0)
            {
                return new OpenCVForUnity.CoreModule.Rect(0, 0, 1, 1);
            }

            return intersection;
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Apply initialization data to instance fields.
        /// </summary>
        /// <param name="handDetector">Hand detector model asset</param>
        /// <param name="handLandmarker">Hand landmarker model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestHandOnly">Whether to use only the best hand</param>
        private void ApplyInitializationData(ModelAsset handDetector, ModelAsset handLandmarker, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestHandOnly)
        {
            if (handDetector != null)
                _handDetector = handDetector;
            if (handLandmarker != null)
                _handLandmarker = handLandmarker;
            if (anchorsCSV != null)
                _anchorsCSV = anchorsCSV;
            _iouThreshold = iouThreshold;
            _scoreThreshold = scoreThreshold;
            _useBestHandOnly = useBestHandOnly;
        }

        /// <summary>
        /// Processes hand detection results and creates EstimationData array
        /// </summary>
        /// <param name="outputIdx">Output indices tensor</param>
        /// <param name="outputScore">Output scores tensor</param>
        /// <param name="outputBox">Output boxes tensor</param>
        /// <param name="M">Affine transformation matrix</param>
        /// <param name="texture">Input texture for landmarker processing</param>
        /// <returns>Array of hand detection results</returns>
        private EstimationData[] ProcessHandDetectionResults(Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox, float2x3 M, Texture texture)
        {
            if (_useBestHandOnly){
                var scorePassesThreshold = outputScore[0] >= _scoreThreshold;

                if (!scorePassesThreshold)
                    return new EstimationData[0];

            }

            var numHands = outputIdx.shape.length;

            // Create EstimationData for the number of detected hands
            EstimationData[] estimationData = new EstimationData[numHands];

            for (int p = 0; p < estimationData.Length; p++)
            {
                var idx = outputIdx[p];

                var (M2, rotation) = ProcessHandLandmarkerPreparation(idx, outputBox, M, texture);

                // Get tensors synchronously
                var (landmarks_screen, confidence, handedness, landmarks_world) = ReadHandLandmarkerTensors();

                using (landmarks_screen)
                using (confidence)
                using (handedness)
                using (landmarks_world)
                {
                    var (landmarks_screen_vec3f, landmarks_world_vec3f, bbox) = ProcessHandLandmarksAndBoundingBox(landmarks_screen, landmarks_world, M2, rotation);

                    estimationData[p] = new EstimationData((float)bbox.tl().x, (float)bbox.tl().y, (float)bbox.br().x, (float)bbox.br().y, landmarks_screen_vec3f, landmarks_world_vec3f, handedness[0], confidence[0]);
                }
            }

            return estimationData;
        }

        /// <summary>
        /// Processes hand detection results asynchronously and creates EstimationData array
        /// </summary>
        /// <param name="outputIdx">Output indices tensor</param>
        /// <param name="outputScore">Output scores tensor</param>
        /// <param name="outputBox">Output boxes tensor</param>
        /// <param name="M">Affine transformation matrix</param>
        /// <param name="texture">Input texture for landmarker processing</param>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Array of hand detection results</returns>
        private async Awaitable<EstimationData[]> ProcessHandDetectionResultsAsync(Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox, float2x3 M, Texture texture, CancellationToken cancellationToken)
        {
            // var scorePassesThreshold = outputScore[0] >= scoreThreshold;

            // if (!scorePassesThreshold)
            //     return new EstimationData[0];

            var numHands = outputIdx.shape.length;

            // Create EstimationData for the number of detected hands
            EstimationData[] estimationData = new EstimationData[numHands];

            for (int p = 0; p < estimationData.Length; p++)
            {
                var idx = outputIdx[p];

                var (M2, rotation) = ProcessHandLandmarkerPreparation(idx, outputBox, M, texture);

                // Get tensors asynchronously
                var (landmarks_screen, confidence, handedness, landmarks_world) = await ReadHandLandmarkerTensorsAsync(cancellationToken);

                using (landmarks_screen)
                using (confidence)
                using (handedness)
                using (landmarks_world)
                {
                    var (landmarks_screen_vec3f, landmarks_world_vec3f, bbox) = ProcessHandLandmarksAndBoundingBox(landmarks_screen, landmarks_world, M2, rotation);

                    estimationData[p] = new EstimationData((float)bbox.tl().x, (float)bbox.tl().y, (float)bbox.br().x, (float)bbox.br().y, landmarks_screen_vec3f, landmarks_world_vec3f, handedness[0], confidence[0]);
                }
            }

            return estimationData;
        }

        /// <summary>
        /// Disposes GPU resources and cleans up references
        /// </summary>
        private void DisposeResources()
        {
            //Debug.LogWarning("_Dispose() called");

            // Ensure GPU resources are released
            _handDetectorWorker?.Dispose();
            _handDetectorWorker = null;
            _handLandmarkerWorker?.Dispose();
            _handLandmarkerWorker = null;
            _detectorInput?.Dispose();
            _detectorInput = null;
            _landmarkerInput?.Dispose();
            _landmarkerInput = null;
        }

        #endregion


        #region Static Methods

        /// <summary>
        /// Visualize hand detection results on Mat.
        /// </summary>
        /// <param name="image">Image to draw on</param>
        /// <param name="data">Hand detection data</param>
        /// <param name="printResult">Whether to print results to console</param>
        /// <param name="isRGB">Whether image is in RGB format</param>
        public static void Visualize(Mat image, EstimationData[] data, bool printResult = false, bool isRGB = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (data == null || data.Length == 0) return;

            for (int i = 0; i < data.Length; i++)
            {
                ref readonly var d = ref data[i];
                float left = d.X1;
                float top = d.Y1;
                float right = d.X2;
                float bottom = d.Y2;

#if NET_STANDARD_2_1
                ReadOnlySpan<Vec3f> landmarksScreen = d.GetLandmarksScreen();
#else
                Vec3f[] landmarksScreen = d.GetLandmarksScreenArray();
#endif

                float handedness = d.Handedness;
                string handednessText = (handedness <= 0.5f) ? "Left" : "Right";
                float confidence = d.Confidence;

                var lineColor = ((byte, byte, byte, byte))SCALAR_WHITE.ToValueTuple();
                var pointColor = (isRGB) ? ((byte, byte, byte, byte))SCALAR_BLUE.ToValueTuple() : ((byte, byte, byte, byte))SCALAR_RED.ToValueTuple();

                // # draw box
                Imgproc.rectangle(image, (left, top), (right, bottom), SCALAR_GREEN.ToValueTuple(), 2);
                Imgproc.putText(image, confidence.ToString("F3"), (left, top + 12), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, pointColor);
                Imgproc.putText(image, handednessText, (left, top + 24), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, pointColor);

                // # Draw line between each key points
                draw_lines(image, landmarksScreen, lineColor, pointColor, true, 2);
            }

            if (printResult)
            {
                StringBuilder sb = new StringBuilder(1024);

                for (int i = 0; i < data.Length; ++i)
                {
                    ref readonly var d = ref data[i];
                    float left = d.X1;
                    float top = d.Y1;
                    float right = d.X2;
                    float bottom = d.Y2;
                    float handedness = d.Handedness;
                    string handednessText = (handedness <= 0.5f) ? "Left" : "Right";
                    float confidence = d.Confidence;

#if NET_STANDARD_2_1
                    ReadOnlySpan<Vec3f> landmarksScreen = d.GetLandmarksScreen();
                    ReadOnlySpan<Vec3f> landmarksWorld = d.GetLandmarksWorld();
#else
                    Vec3f[] landmarksScreen = d.GetLandmarksScreenArray();
                    Vec3f[] landmarksWorld = d.GetLandmarksWorldArray();
#endif

                    sb.AppendFormat("-----------hand {0}-----------", i + 1);
                    sb.AppendLine();
                    sb.AppendFormat("Confidence: {0:F4}", confidence);
                    sb.AppendLine();
                    sb.AppendFormat("Handedness: {0} ({1:F3})", handednessText, handedness);
                    sb.AppendLine();
                    sb.AppendFormat("Hand Box: ({0:F3}, {1:F3}, {2:F3}, {3:F3})", left, top, right, bottom);
                    sb.AppendLine();
                    sb.Append("Hand LandmarksScreen: ");
                    sb.Append("{");
                    for (int j = 0; j < landmarksScreen.Length; j++)
                    {
                        ref readonly var p = ref landmarksScreen[j];
                        sb.AppendFormat("({0:F3}, {1:F3}, {2:F3})", p.Item1, p.Item2, p.Item3);
                        if (j < landmarksScreen.Length - 1)
                            sb.Append(", ");
                    }
                    sb.Append("}");
                    sb.AppendLine();
                    sb.Append("Hand LandmarksWorld: ");
                    sb.Append("{");
                    for (int j = 0; j < landmarksWorld.Length; j++)
                    {
                        ref readonly var p = ref landmarksWorld[j];
                        sb.AppendFormat("({0:F3}, {1:F3}, {2:F3})", p.Item1, p.Item2, p.Item3);
                        if (j < landmarksWorld.Length - 1)
                            sb.Append(", ");
                    }
                    sb.Append("}");
                    sb.AppendLine();
                }

                Debug.Log(sb.ToString());
            }
        }

        /// <summary>
        /// Draws lines between hand key points
        /// </summary>
        /// <param name="image">The input image to draw on.</param>
        /// <param name="landmarks">Hand landmarks array.</param>
        /// <param name="lineColor">Line color.</param>
        /// <param name="pointColor">Point color.</param>
        /// <param name="is_draw_point">Whether to draw points.</param>
        /// <param name="thickness">Line thickness.</param>
#if NET_STANDARD_2_1
        private static void draw_lines(Mat image, ReadOnlySpan<Vec3f> landmarks, (byte, byte, byte, byte) lineColor, (byte, byte, byte, byte) pointColor, bool is_draw_point = true, int thickness = 2)
#else
        private static void draw_lines(Mat image, Vec3f[] landmarks, (byte, byte, byte, byte) lineColor, (byte, byte, byte, byte) pointColor, bool is_draw_point = true, int thickness = 2)
#endif
        {

            // Get key points
            ref readonly var wrist = ref landmarks[(int)KeyPoint.Wrist];
            ref readonly var thumb1 = ref landmarks[(int)KeyPoint.Thumb1];
            ref readonly var thumb2 = ref landmarks[(int)KeyPoint.Thumb2];
            ref readonly var thumb3 = ref landmarks[(int)KeyPoint.Thumb3];
            ref readonly var thumb4 = ref landmarks[(int)KeyPoint.Thumb4];
            ref readonly var index1 = ref landmarks[(int)KeyPoint.Index1];
            ref readonly var index2 = ref landmarks[(int)KeyPoint.Index2];
            ref readonly var index3 = ref landmarks[(int)KeyPoint.Index3];
            ref readonly var index4 = ref landmarks[(int)KeyPoint.Index4];
            ref readonly var middle1 = ref landmarks[(int)KeyPoint.Middle1];
            ref readonly var middle2 = ref landmarks[(int)KeyPoint.Middle2];
            ref readonly var middle3 = ref landmarks[(int)KeyPoint.Middle3];
            ref readonly var middle4 = ref landmarks[(int)KeyPoint.Middle4];
            ref readonly var ring1 = ref landmarks[(int)KeyPoint.Ring1];
            ref readonly var ring2 = ref landmarks[(int)KeyPoint.Ring2];
            ref readonly var ring3 = ref landmarks[(int)KeyPoint.Ring3];
            ref readonly var ring4 = ref landmarks[(int)KeyPoint.Ring4];
            ref readonly var pinky1 = ref landmarks[(int)KeyPoint.Pinky1];
            ref readonly var pinky2 = ref landmarks[(int)KeyPoint.Pinky2];
            ref readonly var pinky3 = ref landmarks[(int)KeyPoint.Pinky3];
            ref readonly var pinky4 = ref landmarks[(int)KeyPoint.Pinky4];

            Imgproc.line(image, (wrist.Item1, wrist.Item2), (thumb1.Item1, thumb1.Item2), lineColor, thickness);
            Imgproc.line(image, (thumb1.Item1, thumb1.Item2), (thumb2.Item1, thumb2.Item2), lineColor, thickness);
            Imgproc.line(image, (thumb2.Item1, thumb2.Item2), (thumb3.Item1, thumb3.Item2), lineColor, thickness);
            Imgproc.line(image, (thumb3.Item1, thumb3.Item2), (thumb4.Item1, thumb4.Item2), lineColor, thickness);

            Imgproc.line(image, (wrist.Item1, wrist.Item2), (index1.Item1, index1.Item2), lineColor, thickness);
            Imgproc.line(image, (index1.Item1, index1.Item2), (index2.Item1, index2.Item2), lineColor, thickness);
            Imgproc.line(image, (index2.Item1, index2.Item2), (index3.Item1, index3.Item2), lineColor, thickness);
            Imgproc.line(image, (index3.Item1, index3.Item2), (index4.Item1, index4.Item2), lineColor, thickness);

            Imgproc.line(image, (wrist.Item1, wrist.Item2), (middle1.Item1, middle1.Item2), lineColor, thickness);
            Imgproc.line(image, (middle1.Item1, middle1.Item2), (middle2.Item1, middle2.Item2), lineColor, thickness);
            Imgproc.line(image, (middle2.Item1, middle2.Item2), (middle3.Item1, middle3.Item2), lineColor, thickness);
            Imgproc.line(image, (middle3.Item1, middle3.Item2), (middle4.Item1, middle4.Item2), lineColor, thickness);

            Imgproc.line(image, (wrist.Item1, wrist.Item2), (ring1.Item1, ring1.Item2), lineColor, thickness);
            Imgproc.line(image, (ring1.Item1, ring1.Item2), (ring2.Item1, ring2.Item2), lineColor, thickness);
            Imgproc.line(image, (ring2.Item1, ring2.Item2), (ring3.Item1, ring3.Item2), lineColor, thickness);
            Imgproc.line(image, (ring3.Item1, ring3.Item2), (ring4.Item1, ring4.Item2), lineColor, thickness);

            Imgproc.line(image, (wrist.Item1, wrist.Item2), (pinky1.Item1, pinky1.Item2), lineColor, thickness);
            Imgproc.line(image, (pinky1.Item1, pinky1.Item2), (pinky2.Item1, pinky2.Item2), lineColor, thickness);
            Imgproc.line(image, (pinky2.Item1, pinky2.Item2), (pinky3.Item1, pinky3.Item2), lineColor, thickness);
            Imgproc.line(image, (pinky3.Item1, pinky3.Item2), (pinky4.Item1, pinky4.Item2), lineColor, thickness);

            if (is_draw_point)
            {
                // # z value is relative to WRIST
                for (int i = 0; i < landmarks.Length; i++)
                {
                    ref readonly var p = ref landmarks[i];
                    int r = Mathf.Max((int)(5 - p.Item3 / 5), 0);
                    r = Mathf.Min(r, 14);
                    Imgproc.circle(image, (p.Item1, p.Item2), r, pointColor, -1);
                }
            }
        }

        #endregion
    }
}
