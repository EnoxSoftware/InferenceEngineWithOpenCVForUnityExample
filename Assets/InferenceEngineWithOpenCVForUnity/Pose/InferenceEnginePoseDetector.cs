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

namespace InferenceEngineWithOpenCVForUnity.Pose
{
    /// <summary>
    /// Pose detector using UnityInferenceEngine for pose detection and landmark estimation.
    /// </summary>
    public class InferenceEnginePoseDetector : InferenceEngineManager<InferenceEnginePoseDetector.EstimationData>
    {
        #region Fields

        // 1. Constants
        private const int K_NUM_ANCHORS = 2254;
        private const int K_NUM_KEYPOINTS = 33;
        private const int DETECTOR_INPUT_SIZE = 224;
        private const int LANDMARKER_INPUT_SIZE = 256;

        // 2. Static fields (color definitions from MediaPipePoseEstimator)
        protected static readonly Scalar SCALAR_WHITE = new Scalar(255, 255, 255, 255);
        protected static readonly Scalar SCALAR_RED = new Scalar(0, 0, 255, 255);
        protected static readonly Scalar SCALAR_GREEN = new Scalar(0, 255, 0, 255);
        protected static readonly Scalar SCALAR_BLUE = new Scalar(255, 0, 0, 255);
        protected static readonly Scalar SCALAR_0 = new Scalar(0, 0, 0, 0);

        // 3. Serialized instance fields
        [SerializeField]
        private ModelAsset _poseDetector;
        [SerializeField]
        private ModelAsset _poseLandmarker;
        [SerializeField]
        private TextAsset _anchorsCSV;
        [SerializeField]
        private float _scoreThreshold = 0.75f;
        [SerializeField]
        private float _iouThreshold = 0.5f;
        [SerializeField]
        private bool _useBestPoseOnly = false;

        // 4. Instance fields
        private float[,] _anchors;
        private Worker _poseDetectorWorker;
        private Worker _poseLandmarkerWorker;
        private Tensor<float> _detectorInput;
        private Tensor<float> _landmarkerInput;
        private float _textureWidth;
        private float _textureHeight;

        // 5. Protected fields
        public UnityEvent<EstimationData[]> OnDetectFinished = new UnityEvent<EstimationData[]>(); // Event notification for multiple DetectionData

        // Properties
        public ModelAsset PoseDetector
        {
            get => _poseDetector;
            set => _poseDetector = value;
        }

        public ModelAsset PoseLandmarker
        {
            get => _poseLandmarker;
            set => _poseLandmarker = value;
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

        public bool UseBestPoseOnly
        {
            get => _useBestPoseOnly;
            set => _useBestPoseOnly = value;
        }

        #endregion

        #region Enums

        /// <summary>
        /// Pose key point enumeration.
        /// </summary>
        public enum KeyPoint
        {
            Nose, LeftEyeInner, LeftEye, LeftEyeOuter, RightEyeInner, RightEye, RightEyeOuter, LeftEar, RightEar,
            MouthLeft, MouthRight,
            LeftShoulder, RightShoulder, LeftElbow, RightElbow, LeftWrist, RightWrist, LeftPinky, RightPinky, LeftIndex, RightIndex, LeftThumb, RightThumb,
            LeftHip, RightHip, LeftKnee, RightKnee, LeftAnkle, RightAnkle, LeftHeel, RightHeel, LeftFootIndex, RightFootIndex
        }

        #endregion

        #region Data Structures

        /// <summary>
        /// Screen landmark data structure.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public readonly struct ScreenLandmark
        {
            public readonly float X;
            public readonly float Y;
            public readonly float Z;
            public readonly float Visibility;
            public readonly float Presence;

            public const int ELEMENT_COUNT = 5;
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Initializes a new instance of the ScreenLandmark struct.
            /// </summary>
            /// <param name="x">X coordinate</param>
            /// <param name="y">Y coordinate</param>
            /// <param name="z">Z coordinate</param>
            /// <param name="visibility">Visibility value</param>
            /// <param name="presence">Presence value</param>
            public ScreenLandmark(float x, float y, float z, float visibility, float presence)
            {
                X = x;
                Y = y;
                Z = z;
                Visibility = visibility;
                Presence = presence;
            }

            /// <summary>
            /// Returns a string representation of the ScreenLandmark.
            /// </summary>
            /// <returns>String representation</returns>
            public readonly override string ToString()
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendFormat("X:{0} Y:{1} Z:{2} Visibility:{3} Presence:{4}", X, Y, Z, Visibility, Presence);
                return sb.ToString();
            }
        }

        /// <summary>
        /// World landmark data structure.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public readonly struct WorldLandmark
        {
            public readonly float X;
            public readonly float Y;
            public readonly float Z;

            public const int ELEMENT_COUNT = 3;
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Initializes a new instance of the WorldLandmark struct.
            /// </summary>
            /// <param name="x">X coordinate</param>
            /// <param name="y">Y coordinate</param>
            /// <param name="z">Z coordinate</param>
            public WorldLandmark(float x, float y, float z)
            {
                X = x;
                Y = y;
                Z = z;
            }

            /// <summary>
            /// Returns a string representation of the WorldLandmark.
            /// </summary>
            /// <returns>String representation</returns>
            public readonly override string ToString()
            {
                var sb = new System.Text.StringBuilder();
                sb.AppendFormat("X:{0} Y:{1} Z:{2}", X, Y, Z);
                return sb.ToString();
            }
        }

        // Data Structures

        /// <summary>
        /// Pose detection and landmark estimation data structure.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public readonly struct EstimationData
        {
            public readonly float X1;
            public readonly float Y1;
            public readonly float X2;
            public readonly float Y2;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = LANDMARK_SCREEN_ELEMENT_COUNT)]
            private readonly float[] _rawLandmarksScreen;

            [MarshalAs(UnmanagedType.ByValArray, SizeConst = LANDMARK_WORLD_ELEMENT_COUNT)]
            private readonly float[] _rawLandmarksWorld;

            public readonly float Confidence;

            public const int LANDMARK_COUNT = 33;
            public const int LANDMARK_SCREEN_ELEMENT_COUNT = 5 * LANDMARK_COUNT; // 33 * 5 = 165
            public const int LANDMARK_WORLD_ELEMENT_COUNT = 3 * LANDMARK_COUNT; // 33 * 3 = 99
            public const int ELEMENT_COUNT = 4 + LANDMARK_SCREEN_ELEMENT_COUNT + LANDMARK_WORLD_ELEMENT_COUNT + 1;
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Initializes a new instance of the EstimationData struct.
            /// </summary>
            /// <param name="x1">Bounding box X1 coordinate</param>
            /// <param name="y1">Bounding box Y1 coordinate</param>
            /// <param name="x2">Bounding box X2 coordinate</param>
            /// <param name="y2">Bounding box Y2 coordinate</param>
            /// <param name="confidence">Confidence score</param>
            /// <param name="landmarksScreen">Screen landmarks</param>
            /// <param name="landmarksWorld">World landmarks</param>
            public EstimationData(float x1, float y1, float x2, float y2, float confidence, ScreenLandmark[] landmarksScreen, WorldLandmark[] landmarksWorld)
            {
                if (landmarksScreen == null || landmarksScreen.Length != LANDMARK_COUNT)
                    throw new ArgumentException("landmarksScreen must be a ScreenLandmark[" + LANDMARK_COUNT + "]");
                if (landmarksWorld == null || landmarksWorld.Length != LANDMARK_COUNT)
                    throw new ArgumentException("landmarksWorld must be a WorldLandmark[" + LANDMARK_COUNT + "]");

                X1 = x1;
                Y1 = y1;
                X2 = x2;
                Y2 = y2;
                _rawLandmarksScreen = new float[LANDMARK_SCREEN_ELEMENT_COUNT];
                for (int i = 0; i < landmarksScreen.Length; i++)
                {
                    int offset = i * 5;
                    ref readonly var landmark = ref landmarksScreen[i];
                    _rawLandmarksScreen[offset + 0] = landmark.X;
                    _rawLandmarksScreen[offset + 1] = landmark.Y;
                    _rawLandmarksScreen[offset + 2] = landmark.Z;
                    _rawLandmarksScreen[offset + 3] = landmark.Visibility;
                    _rawLandmarksScreen[offset + 4] = landmark.Presence;
                }
                _rawLandmarksWorld = new float[LANDMARK_WORLD_ELEMENT_COUNT];
                for (int i = 0; i < landmarksWorld.Length; i++)
                {
                    int offset = i * 3;
                    ref readonly var landmark = ref landmarksWorld[i];
                    _rawLandmarksWorld[offset + 0] = landmark.X;
                    _rawLandmarksWorld[offset + 1] = landmark.Y;
                    _rawLandmarksWorld[offset + 2] = landmark.Z;
                }
                Confidence = confidence;
            }

#if NET_STANDARD_2_1

            public readonly ReadOnlySpan<ScreenLandmark> GetLandmarksScreen()
            {
                return MemoryMarshal.Cast<float, ScreenLandmark>(_rawLandmarksScreen.AsSpan());
            }

            public readonly ReadOnlySpan<WorldLandmark> GetLandmarksWorld()
            {
                return MemoryMarshal.Cast<float, WorldLandmark>(_rawLandmarksWorld.AsSpan());
            }

#endif

            public readonly ScreenLandmark[] GetLandmarksScreenArray()
            {
                var result = new ScreenLandmark[LANDMARK_COUNT];
                for (int i = 0; i < LANDMARK_COUNT; i++)
                {
                    int offset = i * 5;
                    result[i] = new ScreenLandmark(_rawLandmarksScreen[offset + 0], _rawLandmarksScreen[offset + 1], _rawLandmarksScreen[offset + 2], _rawLandmarksScreen[offset + 3], _rawLandmarksScreen[offset + 4]);
                }
                return result;
            }

            public readonly WorldLandmark[] GetLandmarksWorldArray()
            {
                var result = new WorldLandmark[LANDMARK_COUNT];
                for (int i = 0; i < LANDMARK_COUNT; i++)
                {
                    int offset = i * 3;
                    result[i] = new WorldLandmark(_rawLandmarksWorld[offset + 0], _rawLandmarksWorld[offset + 1], _rawLandmarksWorld[offset + 2]);
                }
                return result;
            }

            /// <summary>
            /// Returns a string representation of the EstimationData.
            /// </summary>
            /// <returns>String representation</returns>
            public readonly override string ToString()
            {
                var sb = new System.Text.StringBuilder();

                sb.AppendFormat("X1:{0} Y1:{1} X2:{2} Y2:{3} ", X1, Y1, X2, Y2);

                sb.Append("LandmarksScreen:[");
                var landmarksScreen = GetLandmarksScreenArray();
                for (int i = 0; i < landmarksScreen.Length; i++)
                {
                    sb.Append(landmarksScreen[i]);
                    if (i < landmarksScreen.Length - 1) sb.Append(", ");
                }
                sb.Append("] ");

                sb.Append("LandmarksWorld:[");
                var landmarksWorld = GetLandmarksWorldArray();
                for (int i = 0; i < landmarksWorld.Length; i++)
                {
                    sb.Append(landmarksWorld[i]);
                    if (i < landmarksWorld.Length - 1) sb.Append(", ");
                }
                sb.Append("] ");

                sb.AppendFormat("Confidence:{0}", Confidence);

                return sb.ToString();
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the pose detector with specified parameters.
        /// </summary>
        /// <param name="poseDetector">Pose detector model asset</param>
        /// <param name="poseLandmarker">Pose landmarker model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestPoseOnly">Whether to use only the best pose</param>
        public void Initialize(ModelAsset poseDetector, ModelAsset poseLandmarker, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestPoseOnly)
        {
            OnInitialize += () => ApplyInitializationData(poseDetector, poseLandmarker, anchorsCSV, iouThreshold, scoreThreshold, useBestPoseOnly);
            Initialize();
        }

        /// <summary>
        /// Initialize the pose detector asynchronously with specified parameters.
        /// </summary>
        /// <param name="poseDetector">Pose detector model asset</param>
        /// <param name="poseLandmarker">Pose landmarker model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestPoseOnly">Whether to use only the best pose</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Awaitable task</returns>
        public async Awaitable InitializeAsync(ModelAsset poseDetector, ModelAsset poseLandmarker, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestPoseOnly, CancellationToken cancellationToken = default)
        {
            OnInitialize += () => ApplyInitializationData(poseDetector, poseLandmarker, anchorsCSV, iouThreshold, scoreThreshold, useBestPoseOnly);
            await InitializeAsync(cancellationToken);
        }

        /// <summary>
        /// Perform pose detection synchronously with specified texture. Returns EstimationData array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Pose detection results array</returns>
        public override InferenceEnginePoseDetector.EstimationData[] Infer(Texture texture)
        {
            // Get results
            InferenceEnginePoseDetector.EstimationData[] result = base.Infer(texture);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform pose detection synchronously with specified Mat. Returns EstimationData array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <returns>Pose detection results array</returns>
        public override InferenceEnginePoseDetector.EstimationData[] Infer(Mat mat)
        {
            // Get results
            InferenceEnginePoseDetector.EstimationData[] result = base.Infer(mat);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform pose detection asynchronously with specified texture. Returns EstimationData array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Pose detection results array</returns>
        public override async Awaitable<InferenceEnginePoseDetector.EstimationData[]> InferAsync(Texture texture, CancellationToken cancellationToken = default)
        {
            // Get results
            InferenceEnginePoseDetector.EstimationData[] result = await base.InferAsync(texture, cancellationToken);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform pose detection asynchronously with specified Mat. Returns EstimationData array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Pose detection results array</returns>
        public override async Awaitable<InferenceEnginePoseDetector.EstimationData[]> InferAsync(Mat mat, CancellationToken cancellationToken = default)
        {
            // Get results
            InferenceEnginePoseDetector.EstimationData[] result = await base.InferAsync(mat, cancellationToken);

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
        /// Custom initialization for pose detector.
        /// </summary>
        protected override void InitializeCustom()
        {
            //Debug.LogWarning("_Initialize() called");

            LoadAnchors();
            var poseDetectorModel = CreatePoseDetectorModel();
            var poseLandmarkerModel = CreatePoseLandmarkerModel();
            InitializeWorkersAndTensors(poseDetectorModel, poseLandmarkerModel);
        }

        /// <summary>
        /// Custom asynchronous initialization for pose detector.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        protected override void InitializeAsyncCustom(CancellationToken cancellationToken)
        {
            //Debug.LogWarning("_Initialize() called");

            LoadAnchors(cancellationToken);
            var poseDetectorModel = CreatePoseDetectorModel(cancellationToken);
            var poseLandmarkerModel = CreatePoseLandmarkerModel(cancellationToken);
            InitializeWorkersAndTensors(poseDetectorModel, poseLandmarkerModel, cancellationToken);
        }

        /// <summary>
        /// Custom pose detection implementation for texture input.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Pose detection results array</returns>
#if !UNITY_WEBGL
        protected override InferenceEnginePoseDetector.EstimationData[] InferCustom(Texture texture)
        {
            var M = PrepareTextureAndScheduleInference(texture);

            // Get tensors synchronously
            var (outputIdx, outputScore, outputBox) = ReadPoseDetectorTensors();

            using (outputIdx)
            using (outputScore)
            using (outputBox)
            {
                return ProcessPoseDetectionResults(outputIdx, outputScore, outputBox, M, texture);
            }
        }
#else
        protected override InferenceEnginePoseDetector.EstimationData[] InferCustom(Texture texture)
        {
            Debug.LogWarning("Infer is not supported on WebGL platform due to ReadbackAndClone limitations.");
            return null;
        }
#endif

        /// <summary>
        /// Custom asynchronous pose detection implementation for texture input.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Pose detection results array</returns>
        protected override async Awaitable<InferenceEnginePoseDetector.EstimationData[]> InferAsyncCustom(Texture texture, CancellationToken cancellationToken)
        {
            var M = PrepareTextureAndScheduleInference(texture);

            var (outputIdx, outputScore, outputBox) = await ReadPoseDetectorTensorsAsync(cancellationToken);

            using (outputIdx)
            using (outputScore)
            using (outputBox)
            {
                return await ProcessPoseDetectionResultsAsync(outputIdx, outputScore, outputBox, M, texture, cancellationToken);
            }
        }

        /// <summary>
        /// Custom dispose processing for pose detector.
        /// </summary>
        protected override void DisposeCustom()
        {
            DisposeResources();
        }

        /// <summary>
        /// Custom asynchronous dispose processing for pose detector.
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
        /// Creates and configures the pose detector model with post-processing
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>The compiled pose detector model</returns>
        private Model CreatePoseDetectorModel(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var poseDetectorModel = ModelLoader.Load(_poseDetector);

            var graph = new FunctionalGraph();
            var input = graph.AddInput(poseDetectorModel, 0);
            var outputs = Functional.Forward(poseDetectorModel, input);
            var boxes = outputs[0]; // (1, 2254, 12)
            var scores = outputs[1]; // (1, 2254, 1)

            if (_useBestPoseOnly)
            {
                // post process the model to filter scores + argmax select the best pose
                var idx_scores_boxes = BlazeUtils.ArgMaxFiltering(boxes, scores);
                return graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);
            }
            else
            {
                // post process the model to filter scores + nms select the best poses
                var anchorsData = new float[K_NUM_ANCHORS * 4];
                Buffer.BlockCopy(_anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
                var anchors = Functional.Constant(new TensorShape(K_NUM_ANCHORS, 4), anchorsData);
                var idx_scores_boxes = BlazeUtils.NMSFiltering(boxes, scores, anchors, DETECTOR_INPUT_SIZE, _iouThreshold, _scoreThreshold);
                return graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);
            }
        }

        /// <summary>
        /// Creates and configures the pose landmarker model
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>The pose landmarker model</returns>
        private Model CreatePoseLandmarkerModel(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return ModelLoader.Load(_poseLandmarker);
        }

        /// <summary>
        /// Initializes the workers and input tensors
        /// </summary>
        /// <param name="poseDetectorModel">The compiled pose detector model</param>
        /// <param name="poseLandmarkerModel">The pose landmarker model</param>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        private void InitializeWorkersAndTensors(Model poseDetectorModel, Model poseLandmarkerModel, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _poseDetectorWorker = new Worker(poseDetectorModel, BackendType);
            _poseLandmarkerWorker = new Worker(poseLandmarkerModel, BackendType);
            _detectorInput = new Tensor<float>(new TensorShape(1, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3));
            _landmarkerInput = new Tensor<float>(new TensorShape(1, LANDMARKER_INPUT_SIZE, LANDMARKER_INPUT_SIZE, 3));
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Apply initialization data to instance fields.
        /// </summary>
        /// <param name="poseDetector">Pose detector model asset</param>
        /// <param name="poseLandmarker">Pose landmarker model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestPoseOnly">Whether to use only the best pose</param>
        private void ApplyInitializationData(ModelAsset poseDetector, ModelAsset poseLandmarker, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestPoseOnly)
        {
            if (poseDetector != null)
                _poseDetector = poseDetector;
            if (poseLandmarker != null)
                _poseLandmarker = poseLandmarker;
            if (anchorsCSV != null)
                _anchorsCSV = anchorsCSV;
            _iouThreshold = iouThreshold;
            _scoreThreshold = scoreThreshold;
            _useBestPoseOnly = useBestPoseOnly;
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
            _poseDetectorWorker.Schedule(_detectorInput);

            return M;
        }

        /// <summary>
        /// Reads pose detector tensors synchronously
        /// </summary>
        /// <returns>Tuple containing the read pose detector tensors</returns>
        private (Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox) ReadPoseDetectorTensors()
        {
            var outputIdx = (_poseDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndClone();
            var outputScore = (_poseDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndClone();
            var outputBox = (_poseDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndClone();

            return (outputIdx, outputScore, outputBox);
        }

        /// <summary>
        /// Reads pose detector tensors asynchronously with cancellation support
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Tuple containing the read pose detector tensors</returns>
        private async Awaitable<(Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox)> ReadPoseDetectorTensorsAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var outputIdxAwaitable = (_poseDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
            var outputScoreAwaitable = (_poseDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
            var outputBoxAwaitable = (_poseDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

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
                Debug.LogWarning("Pose detector tensor reading was cancelled in ReadPoseDetectorTensorsAsync");
                // Ensure tensors are disposed if cancellation occurred
                outputIdx?.Dispose();
                outputScore?.Dispose();
                outputBox?.Dispose();

                throw;
            }
        }

        /// <summary>
        /// Reads pose landmarker tensors synchronously
        /// </summary>
        /// <returns>Tuple containing the read pose landmarker tensors</returns>
        private (Tensor<float> landmarks, Tensor<float> confidence, Tensor<float> landmarks_world) ReadPoseLandmarkerTensors()
        {
            var landmarks = (_poseLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndClone(); // (1,195)
            var confidence = (_poseLandmarkerWorker.PeekOutput("Identity_1") as Tensor<float>).ReadbackAndClone(); // (1,1)
            var landmarks_world = (_poseLandmarkerWorker.PeekOutput("Identity_4") as Tensor<float>).ReadbackAndClone(); // (1,117)

            return (landmarks, confidence, landmarks_world);
        }

        /// <summary>
        /// Reads pose landmarker tensors asynchronously with cancellation support
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Tuple containing the read pose landmarker tensors</returns>
        private async Awaitable<(Tensor<float> landmarks, Tensor<float> confidence, Tensor<float> landmarks_world)> ReadPoseLandmarkerTensorsAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var landmarksAwaitable = (_poseLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
            var confidenceAwaitable = (_poseLandmarkerWorker.PeekOutput("Identity_1") as Tensor<float>).ReadbackAndCloneAsync();
            var landmarks_worldAwaitable = (_poseLandmarkerWorker.PeekOutput("Identity_4") as Tensor<float>).ReadbackAndCloneAsync();

            Tensor<float> landmarks = null;
            Tensor<float> confidence = null;
            Tensor<float> landmarks_world = null;

            try
            {
                landmarks = await landmarksAwaitable;
                confidence = await confidenceAwaitable;
                landmarks_world = await landmarks_worldAwaitable;
                cancellationToken.ThrowIfCancellationRequested();

                return (landmarks, confidence, landmarks_world);
            }
            catch (OperationCanceledException)
            {
                Debug.LogWarning("Pose landmarker tensor reading was cancelled in ReadPoseLandmarkerTensorsAsync");
                // Ensure tensors are disposed if cancellation occurred
                landmarks?.Dispose();
                confidence?.Dispose();
                landmarks_world?.Dispose();

                throw;
            }
        }

        /// <summary>
        /// Process pose landmarker preparation for a single detection
        /// </summary>
        /// <param name="idx">Anchor index</param>
        /// <param name="outputBox">Output box tensor</param>
        /// <param name="M">Transformation matrix</param>
        /// <param name="texture">Input texture</param>
        /// <returns>Tuple containing M2 matrix and rotation value</returns>
        private (float2x3 M2, float rotation) ProcessPoseLandmarkerPreparation(int idx, Tensor<float> outputBox, float2x3 M, Texture texture)
        {
            // Calculate anchorPosition
            var anchorX = _anchors[idx, 0];
            var anchorY = _anchors[idx, 1];
            var anchorPosition = DETECTOR_INPUT_SIZE * new float2(anchorX, anchorY);

            // Process box coordinates with float4 in batch
            var boxCoordinates = new float4(
                outputBox[0, 0, 0],  // face X
                outputBox[0, 0, 1],  // face Y
                outputBox[0, 0, 2],  // face width
                outputBox[0, 0, 3]   // face height
            );

            var face_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(boxCoordinates.x, boxCoordinates.y));
            var faceTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(boxCoordinates.x + 0.5f * boxCoordinates.z, boxCoordinates.y + 0.5f * boxCoordinates.w));

            // Process keypoint coordinates with float2
            var keypoint1 = new float2(
                outputBox[0, 0, 4 + 2 * 0 + 0],  // kp1 X
                outputBox[0, 0, 4 + 2 * 0 + 1]   // kp1 Y
            );
            var keypoint2 = new float2(
                outputBox[0, 0, 4 + 2 * 1 + 0],  // kp2 X
                outputBox[0, 0, 4 + 2 * 1 + 1]   // kp2 Y
            );

            var kp1_ImageSpace = BlazeUtils.mul(M, anchorPosition + keypoint1);
            var kp2_ImageSpace = BlazeUtils.mul(M, anchorPosition + keypoint2);
            var delta_ImageSpace = kp2_ImageSpace - kp1_ImageSpace;
            var dscale = 1.25f;
            var radius = dscale * math.length(delta_ImageSpace);
            var theta = math.atan2(delta_ImageSpace.y, delta_ImageSpace.x);
            var rotation = 0.5f * Mathf.PI - theta;
            var origin2 = new float2(0.5f * LANDMARKER_INPUT_SIZE, 0.5f * LANDMARKER_INPUT_SIZE);
            var scale2 = radius / (0.5f * LANDMARKER_INPUT_SIZE);
            var M2 = BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.TranslationMatrix(kp1_ImageSpace), BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))), BlazeUtils.RotationMatrix(rotation)), BlazeUtils.TranslationMatrix(-origin2));
            BlazeUtils.SampleImageAffine(texture, _landmarkerInput, M2);

            _poseLandmarkerWorker.Schedule(_landmarkerInput);

            return (M2, rotation);
        }

        /// <summary>
        /// Process pose landmarks and calculate bounding box
        /// </summary>
        /// <param name="landmarks">Screen landmarks tensor</param>
        /// <param name="landmarks_world">World landmarks tensor</param>
        /// <param name="M2">Transformation matrix for screen coordinates</param>
        /// <param name="rotation">Rotation value</param>
        /// <returns>Tuple containing screen landmarks, world landmarks, and bounding box</returns>
        private (ScreenLandmark[] landmarks_screen_screenlandmark, WorldLandmark[] landmarks_world_worldlandmark, OpenCVForUnity.CoreModule.Rect bbox) ProcessPoseLandmarksAndBoundingBox(Tensor<float> landmarks, Tensor<float> landmarks_world, float2x3 M2, float rotation)
        {
            var M3 = BlazeUtils.RotationMatrix(rotation);

            // Vectorize keypoint processing
            ScreenLandmark[] landmarks_screen_screenlandmark = new ScreenLandmark[K_NUM_KEYPOINTS];
            WorldLandmark[] landmarks_world_worldlandmark = new WorldLandmark[K_NUM_KEYPOINTS];

            // Process 33 keypoints with float3
            for (var i = 0; i < K_NUM_KEYPOINTS; i++)
            {
                // Process 3 float values with float3 simultaneously (remaining 2 are processed individually)
                var landmark_screen_data = new float3(
                    landmarks[5 * i + 0],
                    landmarks[5 * i + 1],
                    landmarks[5 * i + 2]
                );
                var landmark_world_data = new float3(
                    landmarks_world[3 * i + 0],
                    landmarks_world[3 * i + 1],
                    landmarks_world[3 * i + 2]
                );

                float3 landmark_screen = BlazeUtils.mul(M2, landmark_screen_data);
                landmarks_screen_screenlandmark[i] = new ScreenLandmark(landmark_screen.x, _textureHeight - landmark_screen.y, landmark_screen.z, landmarks[5 * i + 3], landmarks[5 * i + 4]);

                float3 landmark_world = BlazeUtils.mul(M3, landmark_world_data);
                landmarks_world_worldlandmark[i] = new WorldLandmark(landmark_world.x, landmark_world.y, landmark_world.z);
            }

            // Optimize Vec2f array creation
            Vec2f[] landmarks_screen_vec2f = new Vec2f[landmarks_screen_screenlandmark.Length];
            for (int i = 0; i < landmarks_screen_vec2f.Length; i++)
            {
                landmarks_screen_vec2f[i] = new Vec2f(landmarks_screen_screenlandmark[i].X, landmarks_screen_screenlandmark[i].Y);
            }
            MatOfPoint points = new MatOfPoint(landmarks_screen_vec2f);
            OpenCVForUnity.CoreModule.Rect bbox = Imgproc.boundingRect(points);
            points.Dispose();

            // Enlarge bounding box to 1.3x size
            bbox = EnlargeBoundingBox(bbox, 1.3);

            // Crop bounding box to intersection with texture boundaries
            bbox = ClampBoundingBoxToImage(bbox);

            return (landmarks_screen_screenlandmark, landmarks_world_worldlandmark, bbox);
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

        /// <summary>
        /// Processes pose detection results and creates EstimationData array
        /// </summary>
        /// <param name="outputIdx">Output indices tensor</param>
        /// <param name="outputScore">Output scores tensor</param>
        /// <param name="outputBox">Output boxes tensor</param>
        /// <param name="M">Affine transformation matrix</param>
        /// <param name="texture">Input texture for landmarker processing</param>
        /// <returns>Array of pose detection results</returns>
        private EstimationData[] ProcessPoseDetectionResults(Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox, float2x3 M, Texture texture)
        {

            if (_useBestPoseOnly){
                var scorePassesThreshold = outputScore[0] >= _scoreThreshold;

                if (!scorePassesThreshold)
                    return new EstimationData[0];

            }

            var numPoses = outputIdx.shape.length;

            // Create EstimationData for the number of detected poses
            EstimationData[] estimationData = new EstimationData[numPoses];

            for (int p = 0; p < estimationData.Length; p++)
            {
                var idx = outputIdx[p];

                var (M2, rotation) = ProcessPoseLandmarkerPreparation(idx, outputBox, M, texture);

                // Get tensors synchronously
                var (landmarks, confidence, landmarks_world) = ReadPoseLandmarkerTensors();

                using (landmarks)
                using (confidence)
                using (landmarks_world)
                {
                    var (landmarks_screen_screenlandmark, landmarks_world_worldlandmark, bbox) = ProcessPoseLandmarksAndBoundingBox(landmarks, landmarks_world, M2, rotation);

                    estimationData[p] = new EstimationData((float)bbox.tl().x, (float)bbox.tl().y, (float)bbox.br().x, (float)bbox.br().y, confidence[0], landmarks_screen_screenlandmark, landmarks_world_worldlandmark);
                }
            }

            return estimationData;
        }

        /// <summary>
        /// Processes pose detection results asynchronously and creates EstimationData array
        /// </summary>
        /// <param name="outputIdx">Output indices tensor</param>
        /// <param name="outputScore">Output scores tensor</param>
        /// <param name="outputBox">Output boxes tensor</param>
        /// <param name="M">Affine transformation matrix</param>
        /// <param name="texture">Input texture for landmarker processing</param>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Array of pose detection results</returns>
        private async Awaitable<EstimationData[]> ProcessPoseDetectionResultsAsync(Tensor<int> outputIdx, Tensor<float> outputScore, Tensor<float> outputBox, float2x3 M, Texture texture, CancellationToken cancellationToken)
        {

            if (_useBestPoseOnly){
                var scorePassesThreshold = outputScore[0] >= _scoreThreshold;

                if (!scorePassesThreshold)
                    return new EstimationData[0];

            }

            var numPoses = outputIdx.shape.length;

            // Create EstimationData for the number of detected poses
            EstimationData[] estimationData = new EstimationData[numPoses];

            for (int p = 0; p < estimationData.Length; p++)
            {
                var idx = outputIdx[p];

                var (M2, rotation) = ProcessPoseLandmarkerPreparation(idx, outputBox, M, texture);

                // Get tensors asynchronously
                var (landmarks, confidence, landmarks_world) = await ReadPoseLandmarkerTensorsAsync(cancellationToken);

                using (landmarks)
                using (confidence)
                using (landmarks_world)
                {
                    var (landmarks_screen_screenlandmark, landmarks_world_worldlandmark, bbox) = ProcessPoseLandmarksAndBoundingBox(landmarks, landmarks_world, M2, rotation);

                    estimationData[p] = new EstimationData((float)bbox.tl().x, (float)bbox.tl().y, (float)bbox.br().x, (float)bbox.br().y, confidence[0], landmarks_screen_screenlandmark, landmarks_world_worldlandmark);
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
            _poseDetectorWorker?.Dispose();
            _poseDetectorWorker = null;
            _poseLandmarkerWorker?.Dispose();
            _poseLandmarkerWorker = null;
            _detectorInput?.Dispose();
            _detectorInput = null;
            _landmarkerInput?.Dispose();
            _landmarkerInput = null;
        }

        #endregion


        #region Static Methods

        /// <summary>
        /// Visualize pose detection results on Mat.
        /// </summary>
        /// <param name="image">Image to draw on</param>
        /// <param name="data">Pose detection data</param>
        /// <param name="printResult">Whether to print results to console</param>
        /// <param name="isRGB">Whether image is in RGB format</param>
        public static void Visualize(Mat image, InferenceEnginePoseDetector.EstimationData[] data, bool printResult = false, bool isRGB = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (data == null || data.Length == 0)
                return;

            for (int i = 0; i < data.Length; i++)
            {
                var d = data[i];
                float left = d.X1;
                float top = d.Y1;
                float right = d.X2;
                float bottom = d.Y2;

#if NET_STANDARD_2_1
                ReadOnlySpan<ScreenLandmark> landmarksScreen = d.GetLandmarksScreen();
                ReadOnlySpan<WorldLandmark> landmarksWorld = d.GetLandmarksWorld();
#else
                ScreenLandmark[] landmarksScreen = d.GetLandmarksScreenArray();
                WorldLandmark[] landmarksWorld = d.GetLandmarksWorldArray();
#endif

                float confidence = d.Confidence;

                var lineColor = SCALAR_WHITE.ToValueTuple();
                var pointColor = (isRGB) ? SCALAR_BLUE.ToValueTuple() : SCALAR_RED.ToValueTuple();

                // # draw box
                Imgproc.rectangle(image, (left, top), (right, bottom), SCALAR_GREEN.ToValueTuple(), 2);
                Imgproc.putText(image, confidence.ToString("F3"), (left, top + 12), Imgproc.FONT_HERSHEY_DUPLEX, 0.5, pointColor);

                // # Draw line between each key points
                draw_lines(landmarksScreen, true);

                if (printResult)
                {
                    StringBuilder sb = new StringBuilder(1536);
                    sb.Append("-----------pose-----------");
                    sb.AppendLine();
                    sb.AppendFormat("Confidence: {0:F4}", confidence);
                    sb.AppendLine();
                    sb.AppendFormat("Person Box: ({0:F3}, {1:F3}, {2:F3}, {3:F3})", left, top, right, bottom);
                    sb.AppendLine();
                    sb.Append("Pose LandmarksScreen: ");
                    sb.Append("{");
                    for (int j = 0; j < landmarksScreen.Length; j++)
                    {
                        ref readonly var p = ref landmarksScreen[j];
                        sb.AppendFormat("({0:F3}, {1:F3}, {2:F3}, {3:F3}, {4:F3})", p.X, p.Y, p.Z, p.Visibility, p.Presence);
                        if (j < landmarksScreen.Length - 1)
                            sb.Append(", ");
                    }
                    sb.Append("}");
                    sb.AppendLine();
                    sb.Append("Pose LandmarksWorld: ");
                    sb.Append("{");
                    for (int j = 0; j < landmarksWorld.Length; j++)
                    {
                        ref readonly var p = ref landmarksWorld[j];
                        sb.AppendFormat("({0:F3}, {1:F3}, {2:F3})", p.X, p.Y, p.Z);
                        if (j < landmarksWorld.Length - 1)
                            sb.Append(", ");
                    }
                    sb.Append("}");
                    sb.AppendLine();

                    Debug.Log(sb.ToString());
                }

#if NET_STANDARD_2_1
                void draw_lines(ReadOnlySpan<ScreenLandmark> landmarks, bool is_draw_point = true)
#else
                void draw_lines(ScreenLandmark[] landmarks, bool is_draw_point = true)
#endif
                {
                    float presenceThreshold = 0.8f;

                    void draw_by_presence(in ScreenLandmark p1, in ScreenLandmark p2)
                    {
                        if (p1.Presence >= presenceThreshold && p2.Presence >= presenceThreshold)
                        {
                            Imgproc.line(image, (p1.X, p1.Y), (p2.X, p2.Y), lineColor, 2);
                        }
                    }

                    // Draw line between each key points
                    ref readonly var nose = ref landmarks[(int)KeyPoint.Nose];
                    ref readonly var leftEyeInner = ref landmarks[(int)KeyPoint.LeftEyeInner];
                    ref readonly var leftEye = ref landmarks[(int)KeyPoint.LeftEye];
                    ref readonly var leftEyeOuter = ref landmarks[(int)KeyPoint.LeftEyeOuter];
                    ref readonly var leftEar = ref landmarks[(int)KeyPoint.LeftEar];
                    ref readonly var rightEyeInner = ref landmarks[(int)KeyPoint.RightEyeInner];
                    ref readonly var rightEye = ref landmarks[(int)KeyPoint.RightEye];
                    ref readonly var rightEyeOuter = ref landmarks[(int)KeyPoint.RightEyeOuter];
                    ref readonly var rightEar = ref landmarks[(int)KeyPoint.RightEar];
                    ref readonly var mouthLeft = ref landmarks[(int)KeyPoint.MouthLeft];
                    ref readonly var mouthRight = ref landmarks[(int)KeyPoint.MouthRight];
                    ref readonly var rightShoulder = ref landmarks[(int)KeyPoint.RightShoulder];
                    ref readonly var rightElbow = ref landmarks[(int)KeyPoint.RightElbow];
                    ref readonly var rightWrist = ref landmarks[(int)KeyPoint.RightWrist];
                    ref readonly var rightThumb = ref landmarks[(int)KeyPoint.RightThumb];
                    ref readonly var rightPinky = ref landmarks[(int)KeyPoint.RightPinky];
                    ref readonly var rightIndex = ref landmarks[(int)KeyPoint.RightIndex];
                    ref readonly var leftShoulder = ref landmarks[(int)KeyPoint.LeftShoulder];
                    ref readonly var leftElbow = ref landmarks[(int)KeyPoint.LeftElbow];
                    ref readonly var leftWrist = ref landmarks[(int)KeyPoint.LeftWrist];
                    ref readonly var leftThumb = ref landmarks[(int)KeyPoint.LeftThumb];
                    ref readonly var leftIndex = ref landmarks[(int)KeyPoint.LeftIndex];
                    ref readonly var leftPinky = ref landmarks[(int)KeyPoint.LeftPinky];
                    ref readonly var leftHip = ref landmarks[(int)KeyPoint.LeftHip];
                    ref readonly var rightHip = ref landmarks[(int)KeyPoint.RightHip];
                    ref readonly var rightKnee = ref landmarks[(int)KeyPoint.RightKnee];
                    ref readonly var rightAnkle = ref landmarks[(int)KeyPoint.RightAnkle];
                    ref readonly var rightHeel = ref landmarks[(int)KeyPoint.RightHeel];
                    ref readonly var rightFootIndex = ref landmarks[(int)KeyPoint.RightFootIndex];
                    ref readonly var leftKnee = ref landmarks[(int)KeyPoint.LeftKnee];
                    ref readonly var leftAnkle = ref landmarks[(int)KeyPoint.LeftAnkle];
                    ref readonly var leftFootIndex = ref landmarks[(int)KeyPoint.LeftFootIndex];
                    ref readonly var leftHeel = ref landmarks[(int)KeyPoint.LeftHeel];

                    draw_by_presence(nose, leftEyeInner);
                    draw_by_presence(leftEyeInner, leftEye);
                    draw_by_presence(leftEye, leftEyeOuter);
                    draw_by_presence(leftEyeOuter, leftEar);
                    draw_by_presence(nose, rightEyeInner);
                    draw_by_presence(rightEyeInner, rightEye);
                    draw_by_presence(rightEye, rightEyeOuter);
                    draw_by_presence(rightEyeOuter, rightEar);

                    draw_by_presence(mouthLeft, mouthRight);

                    draw_by_presence(rightShoulder, rightElbow);
                    draw_by_presence(rightElbow, rightWrist);
                    draw_by_presence(rightWrist, rightThumb);
                    draw_by_presence(rightWrist, rightPinky);
                    draw_by_presence(rightWrist, rightIndex);
                    draw_by_presence(rightPinky, rightIndex);

                    draw_by_presence(leftShoulder, leftElbow);
                    draw_by_presence(leftElbow, leftWrist);
                    draw_by_presence(leftWrist, leftThumb);
                    draw_by_presence(leftWrist, leftIndex);
                    draw_by_presence(leftWrist, leftPinky);

                    draw_by_presence(leftShoulder, rightShoulder);
                    draw_by_presence(leftShoulder, leftHip);
                    draw_by_presence(leftHip, rightHip);
                    draw_by_presence(rightHip, rightShoulder);

                    draw_by_presence(rightHip, rightKnee);
                    draw_by_presence(rightKnee, rightAnkle);
                    draw_by_presence(rightAnkle, rightHeel);
                    draw_by_presence(rightAnkle, rightFootIndex);
                    draw_by_presence(rightHeel, rightFootIndex);

                    draw_by_presence(leftHip, leftKnee);
                    draw_by_presence(leftKnee, leftAnkle);
                    draw_by_presence(leftAnkle, leftFootIndex);
                    draw_by_presence(leftAnkle, leftHeel);
                    draw_by_presence(leftHeel, leftFootIndex);

                    if (is_draw_point)
                    {
                        // # z value is relative to HIP, but we use constant to instead
                        const int auxiliary_points_num = 6;
                        for (int j = 0; j < landmarks.Length - auxiliary_points_num; ++j)
                        {
                            ref readonly var landmark = ref landmarks[j];
                            if (landmark.Presence > presenceThreshold)
                                Imgproc.circle(image, (landmark.X, landmark.Y), 2, pointColor, -1);
                        }
                    }
                }
            }
        }

        #endregion
    }
}
