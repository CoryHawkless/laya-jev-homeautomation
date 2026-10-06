"""
Home-automation decision schema for the floor-plan demo.

Matches the Jev/Laya /v1/systemone wire protocol — send this as the `questions`
body alongside an `{utterance}` state.
"""

HOME_QUESTIONS: dict = {
    "intent": {
        "type": "choice",
        "instructions": "What does the user want to do?",
        "criteria": {
            "light_on":    "turn a light on",
            "light_off":   "turn a light off",
            "light_dim":   "change light brightness up or down",
            "music_play":  "start or resume music playback",
            "music_pause": "stop or pause music",
            "music_vol":   "change music volume up or down",
            "climate_set": "change temperature or thermostat",
            "scene":       "activate a preset scene like movie mode or goodnight",
            "unknown":     "the request is not a home-automation command or is unclear",
        },
    },
    "room": {
        "type": "choice",
        "instructions": "Which area of the home does the command target?",
        "criteria": {
            "living_room": "living room, lounge, den, main area",
            "kitchen":     "kitchen",
            "bedroom":     "master bedroom",
            "office":      "home office or study",
            "outside":     "outdoor, garden, patio, yard",
            "whole_home":  "the whole home, everywhere, or no specific room is mentioned",
        },
    },
    "direction": {
        "type": "choice",
        "instructions": "If the command implies a directional change, which way?",
        "criteria": {
            "up":   "increase, louder, brighter, warmer, hotter",
            "down": "decrease, quieter, dimmer, cooler, colder",
            "none": "no directional change requested",
        },
    },
    "confirm": {
        "type": "choice",
        "instructions": "Is the command ambiguous enough that we should confirm before acting?",
        "criteria": {
            "A": "yes, ask the user to confirm before acting",
            "B": "no, the command is clear enough to act on",
        },
    },
}
