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

namespace InferenceEngineWithOpenCVForUnity.Face
{
    /// <summary>
    /// Face detector using UnityInferenceEngine for face detection and landmark estimation.
    /// </summary>
    public class InferenceEngineFaceDetector : InferenceEngineManager<InferenceEngineFaceDetector.EstimationData>
    {
        #region Fields

        // 1. Constants
        private const int K_NUM_ANCHORS = 896;
        private const int K_NUM_KEYPOINTS = 6;
        private const int DETECTOR_INPUT_SIZE = 128;

        // 2. Static fields
        protected static readonly Scalar SCALAR_BLACK = new Scalar(0, 0, 0, 255);
        protected static readonly Scalar BBOX_COLOR = new Scalar(0, 255, 0, 255);
        protected static readonly Scalar[] KEY_POINTS_COLORS = new Scalar[]
        {
            new(0, 0, 255, 255), // right eye (same color as YuNetV2's right eye)
            new(255, 0, 0, 255), // left eye (same color as YuNetV2's left eye)
            new(255, 255, 0, 255), // nose (same color as YuNetV2's nose tip)
            new(0, 255, 255, 255), // mouth (same color as YuNetV2's mouth right)
            new(255, 255, 255, 255), // right eye tragion (same color as YuNetV2's 6th)
            new(0, 255, 0, 255) // left eye tragion (same color as YuNetV2's mouth left)
        };

        // 3. Serialized instance fields
        [SerializeField]
        private ModelAsset _faceDetector;
        [SerializeField]
        private TextAsset _anchorsCSV;
        [SerializeField]
        private float _iouThreshold = 0.3f;
        [SerializeField]
        private float _scoreThreshold = 0.5f;
        [SerializeField]
        private bool _useBestFaceOnly = false;

        // 4. Instance fields
        private float[,] _anchors;
        private Worker _faceDetectorWorker;
        private Tensor<float> _detectorInput;
        private float _textureWidth;
        private float _textureHeight;

        // 5. Protected fields
        public UnityEvent<EstimationData[]> OnDetectFinished = new UnityEvent<EstimationData[]>(); // Event notification for multiple DetectionData

        // Properties
        public ModelAsset FaceDetector
        {
            get => _faceDetector;
            set => _faceDetector = value;
        }

        public TextAsset AnchorsCSV
        {
            get => _anchorsCSV;
            set => _anchorsCSV = value;
        }

        public float IouThreshold
        {
            get => _iouThreshold;
            set => _iouThreshold = value;
        }

        public float ScoreThreshold
        {
            get => _scoreThreshold;
            set => _scoreThreshold = value;
        }

        public bool UseBestFaceOnly
        {
            get => _useBestFaceOnly;
            set => _useBestFaceOnly = value;
        }

        #endregion

        #region Data Structures

        /// <summary>
        /// Face detection and landmark estimation data structure.
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        public readonly struct EstimationData
        {
            // Bounding box
            public readonly float X;
            public readonly float Y;
            public readonly float Width;
            public readonly float Height;

            // Key points
            public readonly Vec2f RightEye;
            public readonly Vec2f LeftEye;
            public readonly Vec2f Nose;
            public readonly Vec2f Mouth;
            public readonly Vec2f RightEyeTragion;
            public readonly Vec2f LeftEyeTragion;

            // Confidence score [0, 1]
            public readonly float Score;

            public const int LANDMARK_VEC2F_COUNT = 6;
            public const int LANDMARK_ELEMENT_COUNT = 2 * LANDMARK_VEC2F_COUNT;
            public const int ELEMENT_COUNT = 4 + LANDMARK_ELEMENT_COUNT + 1;
            public const int DATA_SIZE = ELEMENT_COUNT * 4;

            /// <summary>
            /// Initializes a new instance of the EstimationData struct.
            /// </summary>
            /// <param name="x">Bounding box X coordinate</param>
            /// <param name="y">Bounding box Y coordinate</param>
            /// <param name="width">Bounding box width</param>
            /// <param name="height">Bounding box height</param>
            /// <param name="rightEye">Right eye landmark</param>
            /// <param name="leftEye">Left eye landmark</param>
            /// <param name="nose">Nose landmark</param>
            /// <param name="mouth">Mouth landmark</param>
            /// <param name="rightEyeTragion">Right eye tragion landmark</param>
            /// <param name="leftEyeTragion">Left eye tragion landmark</param>
            /// <param name="score">Confidence score</param>
            public EstimationData(float x, float y, float width, float height, Vec2f rightEye, Vec2f leftEye, Vec2f nose, Vec2f mouth, Vec2f rightEyeTragion, Vec2f leftEyeTragion, float score)
            {
                X = x;
                Y = y;
                Width = width;
                Height = height;
                RightEye = rightEye;
                LeftEye = leftEye;
                Nose = nose;
                Mouth = mouth;
                RightEyeTragion = rightEyeTragion;
                LeftEyeTragion = leftEyeTragion;
                Score = score;
            }

            /// <summary>
            /// Returns a string representation of the EstimationData.
            /// </summary>
            /// <returns>String representation</returns>
            public readonly override string ToString()
            {
                return $"EstimationData(X:{X} Y:{Y} Width:{Width} Height:{Height} RightEye:{RightEye.ToString()} LeftEye:{LeftEye.ToString()} Nose:{Nose.ToString()} Mouth:{Mouth.ToString()} RightEyeTragion:{RightEyeTragion.ToString()} LeftEyeTragion:{LeftEyeTragion.ToString()} Score:{Score})";
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Initialize the face detector with specified parameters.
        /// </summary>
        /// <param name="faceDetector">Face detector model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestFaceOnly">Whether to use only the best face</param>
        public void Initialize(ModelAsset faceDetector, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestFaceOnly)
        {
            OnInitialize += () => ApplyInitializationData(faceDetector, anchorsCSV, iouThreshold, scoreThreshold, useBestFaceOnly);
            Initialize();
        }

        /// <summary>
        /// Initialize the face detector asynchronously with specified parameters.
        /// </summary>
        /// <param name="faceDetector">Face detector model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestFaceOnly">Whether to use only the best face</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Awaitable task</returns>
        public async Awaitable InitializeAsync(ModelAsset faceDetector, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestFaceOnly, CancellationToken cancellationToken = default)
        {
            OnInitialize += () => ApplyInitializationData(faceDetector, anchorsCSV, iouThreshold, scoreThreshold, useBestFaceOnly);
            await InitializeAsync(cancellationToken);
        }

        #endregion

        #region Protected Methods

        /// <summary>
        /// Custom initialization for face detector.
        /// </summary>
        protected override void InitializeCustom()
        {
            //Debug.LogWarning("_Initialize() called");

            LoadAnchors();
            var faceDetectorModel = CreateFaceDetectorModel();
            InitializeWorkerAndTensor(faceDetectorModel);
        }

        /// <summary>
        /// Custom asynchronous initialization for face detector.
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        protected override void InitializeAsyncCustom(CancellationToken cancellationToken)
        {
            //Debug.LogWarning("_Initialize() called");

            LoadAnchors(cancellationToken);
            var faceDetectorModel = CreateFaceDetectorModel(cancellationToken);
            InitializeWorkerAndTensor(faceDetectorModel, cancellationToken);
        }

        /// <summary>
        /// Custom face detection implementation for texture input.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Face detection results array</returns>
#if !UNITY_WEBGL
        protected override InferenceEngineFaceDetector.EstimationData[] InferCustom(Texture texture)
        {
            var M = PrepareTextureAndScheduleInference(texture);

            // Get tensors synchronously
            var (outputIndices, outputScores, outputBoxes) = ReadFaceDetectorTensors();

            using (outputIndices)
            using (outputScores)
            using (outputBoxes)
            {
                return ProcessFaceDetectionResults(outputIndices, outputScores, outputBoxes, M);
            }
        }
#else
        protected override InferenceEngineFaceDetector.EstimationData[] InferCustom(Texture texture)
        {
            Debug.LogWarning("Infer is not supported on WebGL platform due to ReadbackAndClone limitations.");
            return null;
        }
#endif

        /// <summary>
        /// Custom asynchronous face detection implementation for texture input.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Face detection results array</returns>
        protected override async Awaitable<InferenceEngineFaceDetector.EstimationData[]> InferAsyncCustom(Texture texture, CancellationToken cancellationToken)
        {
            var M = PrepareTextureAndScheduleInference(texture);

            var (outputIndices, outputScores, outputBoxes) = await ReadFaceDetectorTensorsAsync(cancellationToken);

            using (outputIndices)
            using (outputScores)
            using (outputBoxes)
            {
                return ProcessFaceDetectionResults(outputIndices, outputScores, outputBoxes, M);
            }
        }

        /// <summary>
        /// Custom dispose processing for face detector.
        /// </summary>
        protected override void DisposeCustom()
        {
            DisposeResources();
        }

        /// <summary>
        /// Custom asynchronous dispose processing for face detector.
        /// </summary>
        protected override void DisposeAsyncCustom()
        {
            DisposeResources();
        }

        #endregion

        #region Private Methods

        /// <summary>
        /// Apply initialization data to instance fields.
        /// </summary>
        /// <param name="faceDetector">Face detector model asset</param>
        /// <param name="anchorsCSV">Anchors CSV text asset</param>
        /// <param name="iouThreshold">IoU threshold for NMS</param>
        /// <param name="scoreThreshold">Score threshold for detection</param>
        /// <param name="useBestFaceOnly">Whether to use only the best face</param>
        private void ApplyInitializationData(ModelAsset faceDetector, TextAsset anchorsCSV, float iouThreshold, float scoreThreshold, bool useBestFaceOnly)
        {
            if (faceDetector != null)
                _faceDetector = faceDetector;
            if (anchorsCSV != null)
                _anchorsCSV = anchorsCSV;
            _iouThreshold = iouThreshold;
            _scoreThreshold = scoreThreshold;
            _useBestFaceOnly = useBestFaceOnly;
        }

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
        /// Creates and configures the face detector model with post-processing
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>The compiled face detector model</returns>
        private Model CreateFaceDetectorModel(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var faceDetectorModel = ModelLoader.Load(_faceDetector);

            var graph = new FunctionalGraph();
            var input = graph.AddInput(faceDetectorModel, 0);
            var outputs = Functional.Forward(faceDetectorModel, 2 * input - 1);
            var boxes = outputs[0]; // (1, 896, 16)
            var scores = outputs[1]; // (1, 896, 1)

            if (_useBestFaceOnly)
            {
                // post process the model to filter scores + argmax select the best face
                var idx_scores_boxes = BlazeUtils.ArgMaxFiltering(boxes, scores);
                return graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);
            }
            else
            {
                // post process the model to filter scores + nms select the best faces
                var anchorsData = new float[K_NUM_ANCHORS * 4];
                Buffer.BlockCopy(_anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
                var anchors = Functional.Constant(new TensorShape(K_NUM_ANCHORS, 4), anchorsData);
                var idx_scores_boxes = BlazeUtils.NMSFiltering(boxes, scores, anchors, DETECTOR_INPUT_SIZE, _iouThreshold, _scoreThreshold);
                return graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);
            }
        }

        /// <summary>
        /// Initializes the worker and input tensor
        /// </summary>
        /// <param name="faceDetectorModel">The compiled face detector model</param>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        private void InitializeWorkerAndTensor(Model faceDetectorModel, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _faceDetectorWorker = new Worker(faceDetectorModel, BackendType);
            _detectorInput = new Tensor<float>(new TensorShape(1, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, 3));
        }



        /// <summary>
        /// Perform face detection synchronously with specified texture. Returns EstimationData array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <returns>Face detection results array</returns>
        public override InferenceEngineFaceDetector.EstimationData[] Infer(Texture texture)
        {
            // Get results
            InferenceEngineFaceDetector.EstimationData[] result = base.Infer(texture);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform face detection synchronously with specified Mat. Returns EstimationData array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <returns>Face detection results array</returns>
        public override InferenceEngineFaceDetector.EstimationData[] Infer(Mat mat)
        {
            // Get results
            InferenceEngineFaceDetector.EstimationData[] result = base.Infer(mat);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform face detection asynchronously with specified texture. Returns EstimationData array.
        /// </summary>
        /// <param name="texture">Input texture</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Face detection results array</returns>
        public override async Awaitable<InferenceEngineFaceDetector.EstimationData[]> InferAsync(Texture texture, CancellationToken cancellationToken = default)
        {
            // Get results
            InferenceEngineFaceDetector.EstimationData[] result = await base.InferAsync(texture, cancellationToken);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
        }

        /// <summary>
        /// Perform face detection asynchronously with specified Mat. Returns EstimationData array.
        /// </summary>
        /// <param name="mat">Input Mat</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Face detection results array</returns>
        public override async Awaitable<InferenceEngineFaceDetector.EstimationData[]> InferAsync(Mat mat, CancellationToken cancellationToken = default)
        {
            // Get results
            InferenceEngineFaceDetector.EstimationData[] result = await base.InferAsync(mat, cancellationToken);

            // Return null if result is null
            if (result == null)
                return null;

            // Fire UnityEvent (call handlers registered in inspector)
            OnDetectFinished?.Invoke(result);
            return result;
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
            _faceDetectorWorker.Schedule(_detectorInput);

            return M;
        }

        /// <summary>
        /// Processes face detection results and creates EstimationData array
        /// </summary>
        /// <param name="outputIndices">Output indices tensor</param>
        /// <param name="outputScores">Output scores tensor</param>
        /// <param name="outputBoxes">Output boxes tensor</param>
        /// <param name="M">Affine transformation matrix</param>
        /// <returns>Array of face detection results</returns>
        private EstimationData[] ProcessFaceDetectionResults(Tensor<int> outputIndices, Tensor<float> outputScores, Tensor<float> outputBoxes, float2x3 M)
        {
            if (_useBestFaceOnly){
                var scorePassesThreshold = outputScores[0, 0, 0] >= _scoreThreshold;

                if (!scorePassesThreshold)
                    return new EstimationData[0];

            }

            var numFaces = outputIndices.shape.length;

            // Create EstimationData for the number of detected faces
            EstimationData[] estimationData = new EstimationData[numFaces];

            // Process each face sequentially
            for (var i = 0; i < estimationData.Length; i++)
            {
                var idx = outputIndices[i];

                // Reduce duplicate calculations: calculate anchorPosition only once
                var anchorX = _anchors[idx, 0];
                var anchorY = _anchors[idx, 1];
                var anchorPosition = DETECTOR_INPUT_SIZE * new float2(anchorX, anchorY);

                // Reduce duplicate calculations: optimize box_ImageSpace calculation
                var boxOffsetX = outputBoxes[0, i, 0];
                var boxOffsetY = outputBoxes[0, i, 1];
                var boxWidth = outputBoxes[0, i, 2];
                var boxHeight = outputBoxes[0, i, 3];

                // Optimize calculations: reuse intermediate variables
                var boxCenterX = boxOffsetX + 0.5f * boxWidth;
                var boxCenterY = boxOffsetY + 0.5f * boxHeight;

                var box_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(boxOffsetX, boxOffsetY));
                var boxTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(boxCenterX, boxCenterY));

                var boxSize = 2f * (boxTopRight_ImageSpace - box_ImageSpace);

                // Align with OpenCV coordinate system (calculation optimization)
                var boxSizeXHalf = boxSize.x / 2;
                var boxSizeYHalf = boxSize.y / 2;
                Vec2f xy = new Vec2f(box_ImageSpace.x - boxSizeXHalf, (_textureHeight - box_ImageSpace.y) + boxSizeYHalf);
                Vec2f wh = new Vec2f(boxSize.x, -boxSize.y);

                // Optimize array allocation: specify size in advance
                Vec2f[] landmarks_screen = new Vec2f[K_NUM_KEYPOINTS];

                // Vectorize keypoint calculations
                // Process 6 keypoints with 2 float4s (last 2 are unused)
                var keypointOffsetsX = new float4(
                    outputBoxes[0, i, 4 + 2 * 0 + 0],  // keypoint 0 X
                    outputBoxes[0, i, 4 + 2 * 1 + 0],  // keypoint 1 X
                    outputBoxes[0, i, 4 + 2 * 2 + 0],  // keypoint 2 X
                    outputBoxes[0, i, 4 + 2 * 3 + 0]   // keypoint 3 X
                );
                var keypointOffsetsY = new float4(
                    outputBoxes[0, i, 4 + 2 * 0 + 1],  // keypoint 0 Y
                    outputBoxes[0, i, 4 + 2 * 1 + 1],  // keypoint 1 Y
                    outputBoxes[0, i, 4 + 2 * 2 + 1],  // keypoint 2 Y
                    outputBoxes[0, i, 4 + 2 * 3 + 1]   // keypoint 3 Y
                );

                // Process first 4 keypoints
                for (var j = 0; j < 4; j++)
                {
                    var keypointOffsetX = keypointOffsetsX[j];
                    var keypointOffsetY = keypointOffsetsY[j];
                    var position_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(keypointOffsetX, keypointOffsetY));

                    // Align with OpenCV coordinate system (Y coordinate inversion)
                    landmarks_screen[j] = new Vec2f(position_ImageSpace.x, _textureHeight - position_ImageSpace.y);
                }

                // Process remaining 2 keypoints with float2
                var remainingKeypointsX = new float2(
                    outputBoxes[0, i, 4 + 2 * 4 + 0],  // keypoint 4 X
                    outputBoxes[0, i, 4 + 2 * 5 + 0]   // keypoint 5 X
                );
                var remainingKeypointsY = new float2(
                    outputBoxes[0, i, 4 + 2 * 4 + 1],  // keypoint 4 Y
                    outputBoxes[0, i, 4 + 2 * 5 + 1]   // keypoint 5 Y
                );

                for (var j = 4; j < K_NUM_KEYPOINTS; j++)
                {
                    var keypointOffsetX = remainingKeypointsX[j - 4];
                    var keypointOffsetY = remainingKeypointsY[j - 4];
                    var position_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(keypointOffsetX, keypointOffsetY));

                    // Align with OpenCV coordinate system (Y coordinate inversion)
                    landmarks_screen[j] = new Vec2f(position_ImageSpace.x, _textureHeight - position_ImageSpace.y);
                }

                float score = outputScores[0, i, 0];

                estimationData[i] = new EstimationData(xy.Item1, xy.Item2, wh.Item1, wh.Item2, landmarks_screen[0], landmarks_screen[1], landmarks_screen[2], landmarks_screen[3], landmarks_screen[4], landmarks_screen[5], score);
            }
            return estimationData;
        }

        /// <summary>
        /// Reads face detector tensors synchronously
        /// </summary>
        /// <returns>Tuple containing the read face detector tensors</returns>
        private (Tensor<int> outputIndices, Tensor<float> outputScores, Tensor<float> outputBoxes) ReadFaceDetectorTensors()
        {
            var outputIndices = (_faceDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndClone();
            var outputScores = (_faceDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndClone();
            var outputBoxes = (_faceDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndClone();

            return (outputIndices, outputScores, outputBoxes);
        }

        /// <summary>
        /// Reads face detector tensors asynchronously with cancellation support
        /// </summary>
        /// <param name="cancellationToken">Cancellation token for async operations</param>
        /// <returns>Tuple containing the read face detector tensors</returns>
        private async Awaitable<(Tensor<int> outputIndices, Tensor<float> outputScores, Tensor<float> outputBoxes)> ReadFaceDetectorTensorsAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var outputIndicesAwaitable = (_faceDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
            var outputScoresAwaitable = (_faceDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
            var outputBoxesAwaitable = (_faceDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

            Tensor<int> outputIndices = null;
            Tensor<float> outputScores = null;
            Tensor<float> outputBoxes = null;

            try
            {
                outputIndices = await outputIndicesAwaitable;
                outputScores = await outputScoresAwaitable;
                outputBoxes = await outputBoxesAwaitable;
                cancellationToken.ThrowIfCancellationRequested();

                return (outputIndices, outputScores, outputBoxes);
            }
            catch (OperationCanceledException)
            {
                Debug.LogWarning("Face detector tensor reading was cancelled in ReadFaceDetectorTensorsAsync");
                // Ensure tensors are disposed if cancellation occurred
                outputIndices?.Dispose();
                outputScores?.Dispose();
                outputBoxes?.Dispose();

                throw;
            }
        }

        /// <summary>
        /// Disposes GPU resources and cleans up references
        /// </summary>
        private void DisposeResources()
        {
            //Debug.LogWarning("_Dispose() called");

            // Ensure GPU resources are released
            _faceDetectorWorker?.Dispose();
            _faceDetectorWorker = null;
            _detectorInput?.Dispose();
            _detectorInput = null;
        }

        #endregion

        #region Static Methods

        /// <summary>
        /// Visualize InferenceEngineFaceDetector.EstimationData results on Mat.
        /// </summary>
        /// <param name="image">Image to draw on</param>
        /// <param name="data">Face detection data</param>
        /// <param name="printResult">Whether to print results to console</param>
        /// <param name="isRGB">Whether image is in RGB format</param>
        public static void Visualize(Mat image, EstimationData[] data, bool printResult = false, bool isRGB = false)
        {
            if (image != null) image.ThrowIfDisposed();
            if (data == null || data.Length == 0) return;

            for (int i = 0; i < data.Length; i++)
            {
                ref readonly var d = ref data[i];
                float left = d.X;
                float top = d.Y;
                float right = d.X + d.Width;
                float bottom = d.Y + d.Height;
                float score = d.Score;

                var bbc = BBOX_COLOR.ToValueTuple();
                var bbcolor = isRGB ? bbc : (bbc.v2, bbc.v1, bbc.v0, bbc.v3);

                Imgproc.rectangle(image, (left, top), (right, bottom), bbcolor, 2);

                string label = score.ToString("F4");

                int[] baseLine = new int[1];
                var labelSize = Imgproc.getTextSizeAsValueTuple(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine);

                top = Mathf.Max((float)top, (float)labelSize.height);
                Imgproc.rectangle(image, (left, top - labelSize.height),
                    (left + labelSize.width, top + baseLine[0]), bbcolor, Core.FILLED);
                Imgproc.putText(image, label, (left, top), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_BLACK.ToValueTuple(), 1, Imgproc.LINE_AA);

                // draw landmark points
                Imgproc.circle(image, (d.RightEye.Item1, d.RightEye.Item2), 2,
                    isRGB ? KEY_POINTS_COLORS[0].ToValueTuple() : (KEY_POINTS_COLORS[0].val[2], KEY_POINTS_COLORS[0].val[1], KEY_POINTS_COLORS[0].val[0], KEY_POINTS_COLORS[0].val[3]), 2);
                Imgproc.circle(image, (d.LeftEye.Item1, d.LeftEye.Item2), 2,
                    isRGB ? KEY_POINTS_COLORS[1].ToValueTuple() : (KEY_POINTS_COLORS[1].val[2], KEY_POINTS_COLORS[1].val[1], KEY_POINTS_COLORS[1].val[0], KEY_POINTS_COLORS[1].val[3]), 2);
                Imgproc.circle(image, (d.Nose.Item1, d.Nose.Item2), 2,
                    isRGB ? KEY_POINTS_COLORS[2].ToValueTuple() : (KEY_POINTS_COLORS[2].val[2], KEY_POINTS_COLORS[2].val[1], KEY_POINTS_COLORS[2].val[0], KEY_POINTS_COLORS[2].val[3]), 2);
                Imgproc.circle(image, (d.Mouth.Item1, d.Mouth.Item2), 2,
                    isRGB ? KEY_POINTS_COLORS[3].ToValueTuple() : (KEY_POINTS_COLORS[3].val[2], KEY_POINTS_COLORS[3].val[1], KEY_POINTS_COLORS[3].val[0], KEY_POINTS_COLORS[3].val[3]), 2);
                Imgproc.circle(image, (d.RightEyeTragion.Item1, d.RightEyeTragion.Item2), 2,
                    isRGB ? KEY_POINTS_COLORS[4].ToValueTuple() : (KEY_POINTS_COLORS[4].val[2], KEY_POINTS_COLORS[4].val[1], KEY_POINTS_COLORS[4].val[0], KEY_POINTS_COLORS[4].val[3]), 2);
                Imgproc.circle(image, (d.LeftEyeTragion.Item1, d.LeftEyeTragion.Item2), 2,
                    isRGB ? KEY_POINTS_COLORS[5].ToValueTuple() : (KEY_POINTS_COLORS[5].val[2], KEY_POINTS_COLORS[5].val[1], KEY_POINTS_COLORS[5].val[0], KEY_POINTS_COLORS[5].val[3]), 2);
            }

            if (printResult)
            {
                StringBuilder sb = new StringBuilder(128);

                for (int i = 0; i < data.Length; ++i)
                {
                    ref readonly var d = ref data[i];
                    float left = d.X;
                    float top = d.Y;
                    float right = d.X + d.Width;
                    float bottom = d.Y + d.Height;
                    float score = d.Score;

                    sb.AppendFormat("-----------face {0}-----------", i + 1);
                    sb.AppendLine();
                    sb.AppendFormat("Score: {0:F4}", score);
                    sb.AppendLine();
                    sb.AppendFormat("Box: ({0:F3}, {1:F3}, {2:F3}, {3:F3})", left, top, right, bottom);
                    sb.AppendLine();
                    sb.Append("Landmarks: ");
                    sb.Append("{");
                    sb.AppendFormat("({0:F3}, {1:F3}), ", d.RightEye.Item1, d.RightEye.Item2);
                    sb.AppendFormat("({0:F3}, {1:F3}), ", d.LeftEye.Item1, d.LeftEye.Item2);
                    sb.AppendFormat("({0:F3}, {1:F3}), ", d.Nose.Item1, d.Nose.Item2);
                    sb.AppendFormat("({0:F3}, {1:F3}), ", d.Mouth.Item1, d.Mouth.Item2);
                    sb.AppendFormat("({0:F3}, {1:F3}), ", d.RightEyeTragion.Item1, d.RightEyeTragion.Item2);
                    sb.AppendFormat("({0:F3}, {1:F3})", d.LeftEyeTragion.Item1, d.LeftEyeTragion.Item2);
                    sb.Append("}");
                    sb.AppendLine();
                }

                Debug.Log(sb.ToString());
            }
        }

        #endregion
    }
}
