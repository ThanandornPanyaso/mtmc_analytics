# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import logging
from typing import Dict, List, Set, Optional
from datetime import datetime

from mdx.mtmc.schema import Behavior, MTMCObject, MTMCStateObject


class StateManager:
    """
    Module to manage MTMC state

    ::

        mtmc_state = StateManager()
    """

    def __init__(self) -> None:
        self._global_id: int = 0
        self._map_behavior_key_to_global_ids: Dict[str, Set[str]] = dict()
        self._state: Dict[str, MTMCStateObject] = dict()

    def _update_map_behavior_key_to_global_id(self, mtmc_object: MTMCObject) -> None:
        """
        Updates the state's map from behaviors to global IDs

        :param MTMCObject mtmc_object: MTMC object
        :return: None
        """
        for behavior in mtmc_object.matched:
            behavior_key = behavior.key
            if behavior_key not in self._map_behavior_key_to_global_ids.keys():
                self._map_behavior_key_to_global_ids[behavior_key] = set()
            self._map_behavior_key_to_global_ids[behavior_key].add(mtmc_object.globalId)

    def _update_mtmc_state_object(self, mtmc_state_object: MTMCStateObject, mtmc_object: MTMCObject) -> MTMCStateObject:
        """
        Updates the MTMC state object based on the given MTMC object

        :param MTMCStateObject mtmc_state_object: MTMC state object
        :param MTMCObject mtmc_object: MTMC object
        :return: updated MTMC state object
        :rtype: MTMCStateObject
        """
        if mtmc_object.end > mtmc_state_object.end:
            mtmc_state_object.end = mtmc_object.end
        mtmc_state_object.batchId = mtmc_object.batchId
        for behavior in mtmc_object.matched:
            behavior_key = behavior.key
            if behavior_key not in mtmc_state_object.matchedDict.keys():
                mtmc_state_object.matchedDict[behavior_key] = behavior
            else:
                if behavior.end > mtmc_state_object.matchedDict[behavior_key].end:
                    mtmc_state_object.matchedDict[behavior_key].end = behavior.end
                if int(behavior.endFrame) > int(mtmc_state_object.matchedDict[behavior_key].endFrame):
                    mtmc_state_object.matchedDict[behavior_key].endFrame = behavior.endFrame
            mtmc_state_object.matchedDict[behavior_key].matchedSystemTimestamp = datetime.utcnow()
        return mtmc_state_object

    def _convert_mtmc_object_to_mtmc_state_object(self, mtmc_object: MTMCObject) -> MTMCStateObject:
        """
        Converts MTMC object to MTMC state object

        :param MTMCObject mtmc_object: MTMC object
        :return: MTMC state object
        :rtype: MTMCStateObject
        """
        matched_dict: Dict[str, Behavior] = dict()
        for behavior in mtmc_object.matched:
            matched_dict[behavior.key] = behavior

        return MTMCStateObject(
            batchId=mtmc_object.batchId,
            globalId=mtmc_object.globalId,
            objectType=mtmc_object.objectType,
            timestamp=mtmc_object.timestamp,
            end=mtmc_object.end,
            matchedDict=matched_dict
        )

    def _convert_mtmc_state_object_to_mtmc_object(self, mtmc_state_object: MTMCStateObject) -> MTMCObject:
        """
        Converts MTMC state object to MTMC object

        :param MTMCStateObject mtmc_state_object: MTMC state object
        :return: MTMC object
        :rtype: MTMCObject
        """
        matched: List[Behavior] = list()
        for behavior_key in mtmc_state_object.matchedDict.keys():
            matched.append(mtmc_state_object.matchedDict[behavior_key])

        return MTMCObject(
            batchId=mtmc_state_object.batchId,
            globalId=mtmc_state_object.globalId,
            objectType=mtmc_state_object.objectType,
            timestamp=mtmc_state_object.timestamp,
            end=mtmc_state_object.end,
            matched=matched
        )

    def _get_global_ids_assigned_to_matched_behaviors(self, behaviors: List[Behavior]) -> Set[str]:
        """
        Gets a set of global IDs that were previously assigned to matched behaviors

        :param List[Behavior] behaviors: list of behaviors
        :return: set of global IDs
        :rtype: Set[str]
        """
        global_ids: Set[str] = set()
        for behavior in behaviors:
            prev_global_ids = self._map_behavior_key_to_global_ids.get(behavior.key, None)
            if prev_global_ids is not None:
                global_ids = global_ids.union(prev_global_ids)
        return global_ids

    def _get_global_id_with_earliest_timestamp(self, global_ids: Set[str]) -> str:
        """
        Gets the global ID with the earliest timestamp

        :param Set[str] global_ids: set of global IDs
        :return: global ID with the earliest timestamp
        :rtype: str
        """
        map_global_id_to_timestamp = {global_id: self._state[global_id].timestamp for global_id in global_ids}
        chosen_global_id: str = min(map_global_id_to_timestamp, key=map_global_id_to_timestamp.get)
        return chosen_global_id

    def _choose_global_id(self, prev_assigned_global_ids: Set[str], global_ids_used_in_batch: Set[str]) -> Optional[str]:
        """
        Chooses the global ID, given a set of previously assigned global IDs and a set of global IDs that have already been used in the current batch

        :param Set[str] prev_assigned_global_ids: set of previously assigned global IDs
        :param Set[str] global_ids_used_in_batch: set of global IDs that have already been used in the current batch
        :return: chosen global ID or None
        :rtype: Optional[str]
        """
        unused_global_ids = prev_assigned_global_ids.difference(global_ids_used_in_batch)
        if len(unused_global_ids) == 0:
            return None
        elif len(unused_global_ids) == 1:
            (chosen_global_id,) = unused_global_ids
            return chosen_global_id
        else:
            # Since there are more than one unused previously assigned global ID, choose the global ID with the earliest timestamp
            return self._get_global_id_with_earliest_timestamp(unused_global_ids)

    def get_global_ids_in_state(self) -> Set[str]:
        """
        Gets all the global IDs in state

        :return: set of global IDs
        :rtype: Set[str]
        ::

            global_ids = mtmc_state.get_global_ids_in_state()
        """
        return set(self._state.keys())

    def get_mtmc_object(self, global_id: str) -> Optional[MTMCObject]:
        """
        Gets the MTMC object given a global ID

        :param str global_id: global ID
        :return: MTMC object
        :rtype: Optional[MTMCObject]
        ::

            mtmc_object = mtmc_state.get_mtmc_object(global_id)
        """
        mtmc_state_object = self._state.get(global_id, None)
        return None if mtmc_state_object is None else self._convert_mtmc_state_object_to_mtmc_object(mtmc_state_object)

    def get_mtmc_objects(self, global_ids: List[str]) -> List[Optional[MTMCObject]]:
        """
        Gets a list of MTMC objects given a list of global IDs

        :param List[str] global_ids: global IDs
        :return: list of MTMC objects
        :rtype: List[Optional[MTMCObject]]
        ::

            mtmc_objects = mtmc_state.get_mtmc_objects(global_ids)
        """
        return [self.get_mtmc_object(global_id) for global_id in global_ids]

    def get_mtmc_objects_in_state(self) -> List[MTMCObject]:
        """
        Gets all the MTMC objects in state

        :return: list of MTMC objects
        :rtype: List[MTMCObject]
        ::

            mtmc_objects = mtmc_state.get_mtmc_objects_in_state()
        """
        return [self._convert_mtmc_state_object_to_mtmc_object(mtmc_state_object) for mtmc_state_object in self._state.values()]

    def get_map_global_id_to_mtmc_object(self) -> Dict[str, MTMCObject]:
        """
        Gets a map from all the global IDs to MTMC objects in state

        :return: map from global IDs to MTMC objects
        :rtype: Dict[str,MTMCObject]
        ::

            map_global_id_to_mtmc_object = mtmc_state.get_map_global_id_to_mtmc_object()
        """
        return {mtmc_state_object.globalId: self._convert_mtmc_state_object_to_mtmc_object(mtmc_state_object) for mtmc_state_object in self._state.values()}

    def get_updated_mtmc_objects(self, mtmc_objects: List[MTMCObject], batch_id: int) -> List[MTMCObject]:
        """
        Gets updated MTMC objects from state

        :param List[MTMCObject] mtmc_objects: list of MTMC objects
        :param int batch_id: batch ID
        :return: updated MTMC objects
        :rtype: List[MTMCObject]
        ::

            updated_mtmc_objects = mtmc_state.get_updated_mtmc_objects(mtmc_objects, batch_id)
        """
        updated_mtmc_objects: List[MTMCObject] = list()

        global_ids_used_in_batch: Set[str] = set()

        for mtmc_object in sorted(mtmc_objects, key=lambda x: len(x.matched), reverse=True):
            mtmc_object.batchId = str(batch_id)

            # Get global ids which were previously assigned to matched behaviors
            prev_assigned_global_ids = self._get_global_ids_assigned_to_matched_behaviors(mtmc_object.matched)

            # Choose a global ID based on the previous assignment and used ids
            chosen_global_id = self._choose_global_id(prev_assigned_global_ids, global_ids_used_in_batch)

            if chosen_global_id is None:
                # Since no global ID was chosen from prev_assigned_global_ids, assign a new global ID and update state.
                self._global_id += 1
                mtmc_object.globalId = str(self._global_id)
                for i in range(len(mtmc_object.matched)):
                    mtmc_object.matched[i].matchedSystemTimestamp = datetime.utcnow()
                self._state[mtmc_object.globalId] = self._convert_mtmc_object_to_mtmc_state_object(mtmc_object)
                global_ids_used_in_batch.add(mtmc_object.globalId)
                updated_mtmc_objects.append(mtmc_object)
                self._update_map_behavior_key_to_global_id(mtmc_object)
            else:
                # Use the chosen global ID. Update the mtmc state object and the mtmc state
                mtmc_state_object = self._state[chosen_global_id]
                mtmc_state_object = self._update_mtmc_state_object(mtmc_state_object, mtmc_object)
                self._state[chosen_global_id] = mtmc_state_object
                global_ids_used_in_batch.add(chosen_global_id)
                updated_mtmc_object = self._convert_mtmc_state_object_to_mtmc_object(mtmc_state_object)
                updated_mtmc_objects.append(updated_mtmc_object)
                self._update_map_behavior_key_to_global_id(updated_mtmc_object)

        return updated_mtmc_objects

    def delete_older_state(self, keys_in_behavior_state: Set[str]) -> None:
        """
        Deletes global IDs in state when none of the matched behaviors are present in the behavior state

        :param Set[str] keys_in_behavior_state: keys (ID-timestamp pairs) present in the behavior state
        :return: None
        ::

            mtmc_state.delete_older_state(keys_in_behavior_state)
        """
        global_ids_to_be_deleted: Set[str] = set()
        for global_id, mtmc_state_object in self._state.items():
            if keys_in_behavior_state.isdisjoint(set(mtmc_state_object.matchedDict.keys())):
                global_ids_to_be_deleted.add(global_id)

        for global_id in global_ids_to_be_deleted:
            self._state.pop(global_id, None)

        for global_id in global_ids_to_be_deleted:
            for behavior_key in self._map_behavior_key_to_global_ids.keys():
                self._map_behavior_key_to_global_ids[behavior_key].discard(global_id)

        self._map_behavior_key_to_global_ids = {behavior_key: global_ids for behavior_key, global_ids in self._map_behavior_key_to_global_ids.items() if len(global_ids) > 0}

        if len(global_ids_to_be_deleted) > 0:
            logging.info(
                f"No. older global ids removed from state: {len(global_ids_to_be_deleted)}")
