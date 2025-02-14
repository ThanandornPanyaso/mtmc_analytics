# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: schema.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0cschema.proto\x12\x02nv\x1a\x1fgoogle/protobuf/timestamp.proto\"\x82\x01\n\x05\x46rame\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\x12-\n\ttimestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x10\n\x08sensorId\x18\x04 \x01(\t\x12\x1b\n\x07objects\x18\x05 \x03(\x0b\x32\n.nv.Object\"\xf7\x02\n\x06Object\x12\n\n\x02id\x18\x01 \x01(\t\x12\x16\n\x04\x62\x62ox\x18\x02 \x01(\x0b\x32\x08.nv.Bbox\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x12\n\nconfidence\x18\x04 \x01(\x02\x12\"\n\x04info\x18\x05 \x03(\x0b\x32\x14.nv.Object.InfoEntry\x12 \n\tembedding\x18\x06 \x01(\x0b\x32\r.nv.Embedding\x12\x16\n\x04pose\x18\x07 \x01(\x0b\x32\x08.nv.Pose\x12\x16\n\x04gaze\x18\x08 \x01(\x0b\x32\x08.nv.Gaze\x12$\n\x0blipActivity\x18\t \x01(\x0b\x32\x0f.nv.LipActivity\x12\r\n\x05speed\x18\n \x01(\x02\x12\x0b\n\x03\x64ir\x18\x0b \x03(\x02\x12\"\n\ncoordinate\x18\x0c \x01(\x0b\x32\x0e.nv.Coordinate\x12\x1e\n\x08location\x18\r \x01(\x0b\x32\x0c.nv.Location\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"-\n\nCoordinate\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\"1\n\x08Location\x12\x0b\n\x03lat\x18\x01 \x01(\x01\x12\x0b\n\x03lon\x18\x02 \x01(\x01\x12\x0b\n\x03\x61lt\x18\x03 \x01(\x01\"D\n\x04\x42\x62ox\x12\r\n\x05leftX\x18\x01 \x01(\x02\x12\x0c\n\x04topY\x18\x02 \x01(\x02\x12\x0e\n\x06rightX\x18\x03 \x01(\x02\x12\x0f\n\x07\x62ottomY\x18\x04 \x01(\x02\"\xcb\x01\n\x04Pose\x12\x0c\n\x04type\x18\x01 \x01(\t\x12$\n\tkeypoints\x18\x02 \x03(\x0b\x32\x11.nv.Pose.Keypoint\x12 \n\x07\x61\x63tions\x18\x03 \x03(\x0b\x32\x0f.nv.Pose.Action\x1a\x41\n\x08Keypoint\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x13\n\x0b\x63oordinates\x18\x02 \x03(\x02\x12\x12\n\nquaternion\x18\x03 \x03(\x02\x1a*\n\x06\x41\x63tion\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x12\n\nconfidence\x18\x02 \x01(\x02\"C\n\x04Gaze\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\r\n\x05theta\x18\x04 \x01(\x02\x12\x0b\n\x03phi\x18\x05 \x01(\x02\"!\n\x0bLipActivity\x12\x12\n\nclassLabel\x18\x01 \x01(\t\"q\n\x05\x45vent\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12!\n\x04info\x18\x05 \x03(\x0b\x32\x13.nv.Event.InfoEntry\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xad\x01\n\x0f\x41nalyticsModule\x12\n\n\x02id\x18\x01 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x02 \x01(\t\x12\x0e\n\x06source\x18\x03 \x01(\t\x12\x0f\n\x07version\x18\x04 \x01(\t\x12+\n\x04info\x18\x05 \x03(\x0b\x32\x1d.nv.AnalyticsModule.InfoEntry\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xcc\x01\n\x06Sensor\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x03 \x01(\t\x12\x1e\n\x08location\x18\x04 \x01(\x0b\x32\x0c.nv.Location\x12\"\n\ncoordinate\x18\x05 \x01(\x0b\x32\x0e.nv.Coordinate\x12\"\n\x04info\x18\x06 \x03(\x0b\x32\x14.nv.Sensor.InfoEntry\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\xc3\x01\n\x05Place\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x1e\n\x08location\x18\x04 \x01(\x0b\x32\x0c.nv.Location\x12\"\n\ncoordinate\x18\x05 \x01(\x0b\x32\x0e.nv.Coordinate\x12!\n\x04info\x18\x06 \x03(\x0b\x32\x13.nv.Place.InfoEntry\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x8c\x02\n\x07Message\x12\x11\n\tmessageid\x18\x01 \x01(\t\x12\x12\n\nmdsversion\x18\x02 \x01(\t\x12-\n\ttimestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x18\n\x05place\x18\x04 \x01(\x0b\x32\t.nv.Place\x12\x1a\n\x06sensor\x18\x05 \x01(\x0b\x32\n.nv.Sensor\x12,\n\x0f\x61nalyticsModule\x18\x06 \x01(\x0b\x32\x13.nv.AnalyticsModule\x12\x1a\n\x06object\x18\x07 \x01(\x0b\x32\n.nv.Object\x12\x18\n\x05\x65vent\x18\x08 \x01(\x0b\x32\t.nv.Event\x12\x11\n\tvideoPath\x18\t \x01(\t\"s\n\tEmbedding\x12\x12\n\x06vector\x18\x01 \x03(\x02\x42\x02\x10\x01\x12%\n\x04info\x18\x02 \x03(\x0b\x32\x17.nv.Embedding.InfoEntry\x1a+\n\tInfoEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'schema_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _OBJECT_INFOENTRY._options = None
  _OBJECT_INFOENTRY._serialized_options = b'8\001'
  _EVENT_INFOENTRY._options = None
  _EVENT_INFOENTRY._serialized_options = b'8\001'
  _ANALYTICSMODULE_INFOENTRY._options = None
  _ANALYTICSMODULE_INFOENTRY._serialized_options = b'8\001'
  _SENSOR_INFOENTRY._options = None
  _SENSOR_INFOENTRY._serialized_options = b'8\001'
  _PLACE_INFOENTRY._options = None
  _PLACE_INFOENTRY._serialized_options = b'8\001'
  _EMBEDDING_INFOENTRY._options = None
  _EMBEDDING_INFOENTRY._serialized_options = b'8\001'
  _EMBEDDING.fields_by_name['vector']._options = None
  _EMBEDDING.fields_by_name['vector']._serialized_options = b'\020\001'
  _FRAME._serialized_start=54
  _FRAME._serialized_end=184
  _OBJECT._serialized_start=187
  _OBJECT._serialized_end=562
  _OBJECT_INFOENTRY._serialized_start=519
  _OBJECT_INFOENTRY._serialized_end=562
  _COORDINATE._serialized_start=564
  _COORDINATE._serialized_end=609
  _LOCATION._serialized_start=611
  _LOCATION._serialized_end=660
  _BBOX._serialized_start=662
  _BBOX._serialized_end=730
  _POSE._serialized_start=733
  _POSE._serialized_end=936
  _POSE_KEYPOINT._serialized_start=827
  _POSE_KEYPOINT._serialized_end=892
  _POSE_ACTION._serialized_start=894
  _POSE_ACTION._serialized_end=936
  _GAZE._serialized_start=938
  _GAZE._serialized_end=1005
  _LIPACTIVITY._serialized_start=1007
  _LIPACTIVITY._serialized_end=1040
  _EVENT._serialized_start=1042
  _EVENT._serialized_end=1155
  _EVENT_INFOENTRY._serialized_start=519
  _EVENT_INFOENTRY._serialized_end=562
  _ANALYTICSMODULE._serialized_start=1158
  _ANALYTICSMODULE._serialized_end=1331
  _ANALYTICSMODULE_INFOENTRY._serialized_start=519
  _ANALYTICSMODULE_INFOENTRY._serialized_end=562
  _SENSOR._serialized_start=1334
  _SENSOR._serialized_end=1538
  _SENSOR_INFOENTRY._serialized_start=519
  _SENSOR_INFOENTRY._serialized_end=562
  _PLACE._serialized_start=1541
  _PLACE._serialized_end=1736
  _PLACE_INFOENTRY._serialized_start=519
  _PLACE_INFOENTRY._serialized_end=562
  _MESSAGE._serialized_start=1739
  _MESSAGE._serialized_end=2007
  _EMBEDDING._serialized_start=2009
  _EMBEDDING._serialized_end=2124
  _EMBEDDING_INFOENTRY._serialized_start=519
  _EMBEDDING_INFOENTRY._serialized_end=562
# @@protoc_insertion_point(module_scope)
