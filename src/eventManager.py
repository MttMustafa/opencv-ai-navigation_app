import time
import json

tr = open('../messageTemplates/tr.json')
msg_db = json.load(tr)

# keeps the evets
# {"msg_id" : event_time}
event_queue = {}
# keeps the events recorded x amount of time ago
event_time_tracker = {}

#adds events to event_queue
def addEvent(msg_id, event_time):

    # check if event recorded before
    # substract the recorded time from the new event time
    # if not enough time has elapsed do not do anything
    # if it did delete it from time tracker so the event can be added to the list later
    if msg_id in event_time_tracker:
        print('ev gate')
        if (event_time - event_time_tracker[msg_id]) > 5:
            del event_time_tracker[msg_id]

    # if event is not in event queue and itime tracker 
    # add the event to both of them
    if (msg_id not in event_queue) and (msg_id not in event_time_tracker):
        print('msg gate')
        event_time_tracker[msg_id] = event_time
        event_queue[msg_id] = event_time


# gets events
# if event_queue is not empty
# get the id of the event with the minimum time value (time recorded in epoch form)
# delete it from event queue
# find the message from imported language json file (which parsed into hash table) and return it
def getEvent():
    if event_queue:
        id = min(event_queue, key=event_queue.get)
        del event_queue[id]
        return msg_db[id]


