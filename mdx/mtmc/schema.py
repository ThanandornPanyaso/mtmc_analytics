# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
from typing import List, Dict, Set, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel


def convert_datetime_to_iso_8601_with_z_suffix(dt: datetime) -> str:
    """
    Converts datetime to ISO 8601 format with Z suffix

    :param datetime dt: datetime
    :return: string of the datetime
    :rtype: str
    ::

        dt_str = convert_datetime_to_iso_8601_with_z_suffix(dt)
    """
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Bbox(BaseModel):
    leftX: float
    topY: float
    rightX: float
    bottomY: float


class Object(BaseModel):
    id: str
    bbox: Bbox
    type: str
    confidence: float
    info: Optional[Dict[str, str]]
    embedding: Optional[List[float]]


class Frame(BaseModel):
    version: str
    id: str
    timestamp: datetime
    sensorId: str
    objects: List[Object]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class Behavior(BaseModel):
    key: str
    id: str
    sensorId: str
    objectId: str
    objectType: str
    timestamp: datetime
    end: datetime
    startFrame: str
    endFrame: str
    place: str = ""
    matchedSystemTimestamp: Optional[datetime] = None
    timestamps: Optional[List[datetime]]
    frameIds: Optional[List[str]]
    bboxes: Optional[List[Bbox]]
    confidences: Optional[List[float]]
    locations: Optional[List[List[float]]]
    locationMask: Optional[List[bool]]
    embeddings: Optional[List[List[float]]]
    embeddingMask: Optional[List[bool]]
    roiMask: Optional[List[bool]]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class BehaviorStateObjects(BaseModel):
    maxTimestamp: Optional[datetime]
    behaviorDict: Dict[datetime, Behavior]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class MTMCObject(BaseModel):
    batchId: str
    globalId: str
    objectType: str
    timestamp: datetime
    end: datetime
    matched: List[Behavior]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class MTMCStateObject(BaseModel):
    batchId: str
    globalId: str
    objectType: str
    timestamp: datetime
    end: datetime
    matchedDict: Dict[str, Behavior]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class MTMCObjectPlusCount(BaseModel):
    type: str
    count: int


class MTMCObjectPlusLocations(BaseModel):
    id: str
    type: str
    locations: List[List[float]]
    matchedBehaviorKeys: Set[str]


class MTMCObjectsPlus(BaseModel):
    place: str
    timestamp: datetime
    frameId: str
    objectCounts: List[MTMCObjectPlusCount]
    locationsOfObjects: List[MTMCObjectPlusLocations]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class MTMCStateObjectPlus(BaseModel):
    id: str
    type: str
    embeddings: List[List[float]]
    locationsDict: Dict[datetime, List[List[float]]]
    matchedBehaviorKeys: Set[str]

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class Sensor(BaseModel):
    type: str
    id: str
    origin: Dict[str, float]
    geoLocation: Dict[str, float]
    coordinates: Dict[str, float]
    translationToGlobalCoordinates: Dict[str, float] = {"x": 0.0, "y": 0.0}
    scaleFactor: float
    attributes: List[Dict[str, str]]
    place: List[Dict[str, str]]
    imageCoordinates: List[Dict[str, float]]
    globalCoordinates: List[Dict[str, float]]
    intrinsicMatrix: Optional[List[List[float]]] = None
    extrinsicMatrix: Optional[List[List[float]]] = None
    cameraMatrix: Optional[List[List[float]]] = None
    homography: Optional[List[List[float]]] = None
    tripwires: List[Dict[str, Any]] = list()
    rois: List[Dict[str, Any]] = list()


class SensorStateObject(BaseModel):
    placeStr: str = ""
    frameWidth: Optional[int] = None
    frameHeight: Optional[int] = None
    fps: Optional[float] = None
    direction: Optional[float] = None
    fieldOfViewPolygon: Optional[str] = None
    homography: Optional[List[List[float]]] = None
    cameraMatrix: Optional[List[List[float]]] = None
    distortionCoeffs: Optional[List[List[float]]] = None
    cameraPosition: Optional[List[List[float]]] = None
    rotationVector: Optional[List[List[float]]] = None
    translationVector: Optional[List[List[float]]] = None
    rotationMatrix: Optional[List[List[float]]] = None
    eulerAngles: Optional[List[List[float]]] = None
    rois: Optional[List[List[Tuple[float, float]]]] = None
    timestamp: Optional[datetime] = None
    sensor: Sensor

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }


class Notification(BaseModel):
    sensors: Optional[List[Sensor]] = None
    event_type: str
    timestamp: datetime
    message: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: convert_datetime_to_iso_8601_with_z_suffix
        }
