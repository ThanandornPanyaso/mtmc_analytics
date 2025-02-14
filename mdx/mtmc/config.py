# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
from typing import List, Optional
from enum import Enum
from omegaconf import MISSING
from pydantic import BaseModel


class AppIOConfig(BaseModel):
    """
    App config for input and output
    """
    enableDebug: bool = False
    inMtmcPlusBatchMode: bool = False
    batchId: str = MISSING
    selectedSensorIds: List[str] = list()
    outputDirPath: str = MISSING
    videoDirPath: str = MISSING
    jsonDataPath: str = MISSING
    protobufDataPath: str = MISSING
    groundTruthPath: str = MISSING
    groundTruthFrameIdOffset: int = 1
    useFullBodyGroundTruth: bool = False
    use3dEvaluation: bool = False
    plotEvaluationGraphs: bool = False


class AppPreprocessingConfig(BaseModel):
    """
    App config for pre-processing and filtering behaviors
    """
    filterByRegionsOfInterest: bool = False
    timestampThreshMin: Optional[float] = None
    locationBboxBottomGapThresh: float = 0.02
    locationConfidenceThresh: float = 0.5
    locationBboxAreaThresh: float = 0.0008
    locationBboxAspectRatioThresh: float = 0.6
    embeddingBboxBottomGapThresh: float = 0.02
    embeddingConfidenceThresh: float = 0.5
    embeddingBboxAreaThresh: float = 0.0008
    embeddingBboxAspectRatioThresh: float = 0.6
    embeddingVisibilityThresh: float = 0.5
    behaviorConfidenceThresh: float = 0.45
    behaviorBboxAreaThresh: float = 0.0007
    behaviorBboxAspectRatioThresh: float = 0.75
    behaviorLengthThreshSec: float = 0.0
    shortBehaviorFinishThreshSec: Optional[float] = None
    behaviorNumLocationsMax: int = 9000
    behaviorSplitThreshSec: int = 6
    behaviorRetentionInStateSec: float = 600.0
    mtmcPlusRetentionInStateSec: float = 10.0
    mtmcPlusInitBufferLenSec: float = 10.0
    mtmcPlusReinitRatioAssignedBehaviors: float = 0.75
    mtmcPlusReinitDiffRatioClusters: Optional[float] = None


class AppLocalizationConfig(BaseModel):
    """
    App config for localization
    """
    rectifyBboxByCalibration: bool = False
    peopleHeightMaxLengthSec: int = 600
    peopleHeightNumSamplesMax: int = 1000
    peopleHeightNumBatchFrames: int = 10000
    peopleHeightEstimationRatio: float = 0.7
    peopleHeightVisibilityThresh: float = 0.8
    overwrittenPeopleHeightMeter: Optional[float] = None


class ClusteringAlgoEnum(str, Enum):
    AgglomerativeClustering = "AgglomerativeClustering"
    HDBSCAN = "HDBSCAN"


class SpatioTemporalDistTypeEnum(str, Enum):
    Hausdorff = "Hausdorff"
    pairwise = "pairwise"


class AppClusteringConfig(BaseModel):
    """
    App config for clustering
    """
    clusteringAlgo: ClusteringAlgoEnum = ClusteringAlgoEnum.HDBSCAN
    overwrittenNumClusters: Optional[int] = None
    agglomerativeClusteringDistThresh: float = 3.5
    hdbscanMinClusterSize: int = 5
    numReassignmentIterations: int = 4
    reassignmentDistLooseThresh: float = 1.0
    reassignmentDistTightThresh: float = 0.12
    spatioTemporalDistLambda: float = 0.15
    spatioTemporalDistType: SpatioTemporalDistTypeEnum = SpatioTemporalDistTypeEnum.Hausdorff
    spatioTemporalDirMagnitudeThresh: float = 0.5
    enableOnlineSpatioTemporalConstraint: bool = False
    onlineSpatioTemporalDistThresh: Optional[float] = None
    suppressOverlappingBehaviors: bool = False
    meanEmbeddingsUpdateRate: float = 0.1
    skipAssignedBehaviors: bool = True
    enableOnlineDynamicUpdate: bool = True
    dynamicUpdateAppearanceDistThresh: float = 0.2
    dynamicUpdateSpatioTemporalDistThresh: float = 10.0
    dynamicUpdateLengthThreshSec: float = 9.0

class AppStreamingConfig(BaseModel):
    """
    App config for streaming (Kafka)
    """
    kafkaBootstrapServers: str = "localhost:9092"
    kafkaProducerLingerMs: int = 0
    kafkaMicroBatchIntervalSec: float = 60.0
    kafkaRawConsumerPollTimeoutMs: int = 10000
    kafkaNotificationConsumerPollTimeoutMs: int = 100
    kafkaConsumerMaxRecordsPerPoll: int = 100000
    sendEmptyMtmcPlusMessages: bool = True
    mtmcPlusFrameBatchSizeMs: int = 180
    mtmcPlusBehaviorBatchesConsumed: int = 4
    mtmcPlusFrameBufferResetSec: float = 4.0
    mtmcPlusTimestampDelayMs: int = 100
    mtmcPlusLocationWindowSec: float = 1.0
    mtmcPlusSmoothingWindowSec: float = 1.0
    mtmcPlusNumProcessesMax: int = 8


class AppConfig(BaseModel):
    """
    App config
    """
    io: AppIOConfig = AppIOConfig()
    preprocessing: AppPreprocessingConfig = AppPreprocessingConfig()
    localization: AppLocalizationConfig = AppLocalizationConfig()
    clustering: AppClusteringConfig = AppClusteringConfig()
    streaming: AppStreamingConfig = AppStreamingConfig()


class VizMtmcSetupConfig(BaseModel):
    """
    MTMC visualization config for setup
    """
    # [frames, behaviors, mtmc_objects, ground_truth_bboxes, ground_truth_locations]
    vizMode: str = "mtmc_objects"
    # [grid, sequence, topview]
    vizMtmcObjectsMode: str = "grid"
    enableMultiprocessing: bool = False
    ffmpegRequired: bool = False


class VizMtmcIOConfig(BaseModel):
    """
    MTMC visualization config for input and output
    """
    selectedSensorIds: List[str] = list()
    selectedBehaviorIds: List[str] = list()
    selectedGlobalIds: List[str] = list()
    outputDirPath: str = MISSING
    videoDirPath: str = MISSING
    mapPath: str = MISSING
    framesPath: str = MISSING
    behaviorsPath: str = MISSING
    mtmcObjectsPath: str = MISSING
    groundTruthPath: str = MISSING


class VizMtmcPlottingConfig(BaseModel):
    """
    MTMC visualization config for plotting
    """
    gridLayout: List[int] = [2, 2]
    blankOutEmptyFrames: bool = False
    vizFilteredFrames: bool = False
    outputFrameHeight: int = -1
    tailLengthMax: int = 200
    smoothingTailLengthThresh: int = 5
    smoothingTailWindow: int = 30


class VizMtmcConfig(BaseModel):
    """
    MTMC visualization config
    """
    setup: VizMtmcSetupConfig = VizMtmcSetupConfig()
    io: VizMtmcIOConfig = VizMtmcIOConfig()
    plotting: VizMtmcPlottingConfig = VizMtmcPlottingConfig()
    
    
class VizRtlsInputConfig(BaseModel):
    """
    RTLS visualization config for input
    """
    calibrationPath: str = MISSING
    videoDirPath: str = MISSING
    mapPath: str = MISSING
    rtlsLogPath: str = MISSING
    rawDataPath: str = MISSING


class SensorSetupEnum(int, Enum):
    numSensors8 = 8
    numSensors12 = 12
    numSensors16 = 16
    numSensors30 = 30
    numSensors40 = 40
    numSensors96 = 96
    numSensors100 = 100


class SensorViewsLayoutEnum(str, Enum):
    radial = "radial"
    split = "split"


class SensorDisplayModeEnum(str, Enum):
    rotational = "rotational"
    cumulative = "cumulative"


class VizRtlsOutputConfig(BaseModel):
    """
    RTLS visualization config for output
    """
    outputVideoPath: str = MISSING
    outputMapHeight: int = 1080
    displaySensorViews: bool = False
    sensorViewsLayout: SensorViewsLayoutEnum = SensorViewsLayoutEnum.radial
    sensorViewDisplayMode: SensorDisplayModeEnum = SensorDisplayModeEnum.rotational
    sensorFovDisplayMode: SensorDisplayModeEnum = SensorDisplayModeEnum.rotational
    skippedBeginningTimeSec: float = 0.0
    outputVideoDurationSec: float = 60.0
    sensorSetup: SensorSetupEnum = SensorSetupEnum.numSensors30
    bufferLengthThreshSec: float = 3.0
    trajectoryLengthThreshSec: float = 5.0
    sensorViewStartTimeSec: float = 2.0
    sensorViewDurationSec: float = 1.0
    sensorViewGapSec: float = 0.1


class VizRtlsConfig(BaseModel):
    """
    RTLS visualization config
    """
    input: VizRtlsInputConfig = VizRtlsInputConfig()
    output: VizRtlsOutputConfig = VizRtlsOutputConfig()
