# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
import logging
import random
from . import qa_service as qs

logger = logging.getLogger(__name__)

support_search = ["话费", "流量"]

default_answer = ["您说什么，小贝听不懂。","你说的小贝不懂呀！","不好意思，您能换个说法吗？"]

def getDefault_answer():
    index = random.randint(0,len(default_answer))
    return default_answer[index]


def extract_item(item):
    if item is None:
        return None
    for name in support_search:
        if name in item:
            return name
    return None


class ActionSearchConsume(Action):
    def name(self):
        return 'action_search_consume'

    def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        item = tracker.get_slot("item")
        item = extract_item(item)
        if item is None:
            dispatcher.utter_message("您好，我现在只会查话费和流量")
            dispatcher.utter_message("你可以这样问我：“帮我查话费”")
            return []

        time = tracker.get_slot("time")
        if time is None:
            dispatcher.utter_message("您想查询哪个月的消费？")
            return []
        # query database here using item and time as key. but you may normalize time format first.
        dispatcher.utter_message("好，请稍等")
        if item == "流量":
            dispatcher.utter_message(
                "您好，您{}共使用{}二百八十兆，剩余三十兆。".format(time, item))
        else:
            dispatcher.utter_message("您好，您{}共消费二十八元。".format(time))
        return []

class ActionSystemHowTo(Action):
    
    def name(self):
        return 'action_system_how_to'
    
    def run(self,
           dispatcher: CollectingDispatcher,
           tracker: Tracker,
           domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        text = tracker.latest_message['text']
        answer = self.service.getanswer(text)
        if answer:
            dispatcher.utter_message(answer)
        else:
            dispatcher.utter_message(getDefault_answer())
        return []

    def __init__(self):
        self.service = qs.QAService()

