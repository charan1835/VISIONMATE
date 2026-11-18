"""
Natural-language response generator module for VisionMate.

Generates concise, human-friendly text feedback for target object queries
based on parsed command intents, live perception pipeline detections, and query memory history.
"""

from typing import Dict, Any, List, Optional
import random


class ObjectResponseGenerator:
    """
    Deterministic response generator converting perception metadata and user intent
    into natural spoken/written feedback.
    """

    # Human-friendly distance phrases — written as natural speech continuations
    DISTANCE_PHRASES: Dict[str, List[str]] = {
        "far":        ["pretty far away", "quite far from you", "far off"],
        "medium":     ["about mid-range", "not too far from you", "a moderate distance away"],
        "close":      ["fairly close to you", "pretty close by", "not far from you"],
        "very_close": ["right there, very close to you", "almost right in front of you", "extremely close — be careful"],
    }

    # Natural direction phrases — conversational
    # Key: (horizontal, vertical) -> list of phrase options
    POSITION_PHRASES: Dict[tuple, List[str]] = {
        ("left",   "top"):    ["in the upper left corner", "up and to your left", "top left of your view"],
        ("left",   "middle"): ["to your left", "on the left side", "over to your left"],
        ("left",   "bottom"): ["in the lower left corner", "down and to your left", "bottom left of your view"],
        ("center", "top"):    ["above you in the center", "up ahead in the middle", "top center of your view"],
        ("center", "middle"): ["directly ahead of you", "right in front of you", "straight ahead"],
        ("center", "bottom"): ["below in the center", "down in front of you", "lower center of your view"],
        ("right",  "top"):    ["in the upper right corner", "up and to your right", "top right of your view"],
        ("right",  "middle"): ["to your right", "on the right side", "over to your right"],
        ("right",  "bottom"): ["in the lower right corner", "down and to your right", "bottom right of your view"],
    }

    def _get_plural(self, word: str) -> str:
        """Return the plural form of an object label."""
        w = word.strip().lower()
        if w == "person":
            return "people"
        elif w.endswith("y") and not w.endswith(("ay", "ey", "oy", "uy")):
            return w[:-1] + "ies"
        elif w.endswith(("s", "sh", "ch", "x", "z")):
            return w + "es"
        else:
            return w + "s"

    def _format_position(self, pos_dict: Optional[Dict[str, Any]]) -> str:
        """
        Convert 2D spatial position metadata into a conversational natural language phrase.
        e.g. "to your right", "in the upper left corner", "directly ahead of you"
        """
        if not isinstance(pos_dict, dict):
            return "somewhere in view"

        horiz = str(pos_dict.get("horizontal", "center")).lower()
        vert  = str(pos_dict.get("vertical",   "middle")).lower()

        key = (horiz, vert)
        options = self.POSITION_PHRASES.get(key)
        if options:
            return random.choice(options)

        # Fallback for unexpected values
        if horiz == "center" and vert == "middle":
            return "right in front of you"
        return f"to your {horiz}"

    def _format_distance(self, dist_dict: Optional[Dict[str, Any]]) -> str:
        """Convert relative distance metadata into a natural language phrase."""
        if not isinstance(dist_dict, dict):
            return ""
        category = str(dist_dict.get("category", "")).lower()
        options = self.DISTANCE_PHRASES.get(category)
        if options:
            return random.choice(options)
        return ""

    def _build_response(
        self,
        verb: str,
        target: str,
        pos_phrase: str,
        dist_phrase: str,
    ) -> str:
        """
        Build a natural flowing sentence like:
          "Your phone is to your right, pretty close to you."
          "I can see your bottle directly ahead, about mid-range."
        """
        if dist_phrase:
            templates = [
                f"Your {target} is {pos_phrase}, {dist_phrase}.",
                f"I can see your {target} {pos_phrase}, {dist_phrase}.",
                f"Found your {target}. It's {pos_phrase}, {dist_phrase}.",
            ]
        else:
            templates = [
                f"Your {target} is {pos_phrase}.",
                f"I can see your {target} {pos_phrase}.",
                f"Found your {target} {pos_phrase}.",
            ]
        return random.choice(templates)

    def _format_single_match(self, target_object: str, detection: Dict[str, Any]) -> str:
        """Format a single object detection into a natural language sentence."""
        pos_phrase  = self._format_position(detection.get("position"))
        dist_phrase = self._format_distance(detection.get("distance"))
        return self._build_response("found", target_object.lower(), pos_phrase, dist_phrase)

    def _format_multiple_matches(self, target_object: str, matches: List[Dict[str, Any]]) -> str:
        """Format a multi-object response with location detail for the top 2 instances."""
        count  = len(matches)
        plural = self._get_plural(target_object)
        p1 = self._format_position(matches[0].get("position"))
        if count == 2:
            p2 = self._format_position(matches[1].get("position"))
            return f"I can see 2 {plural}. One is {p1} and the other is {p2}."
        else:
            p2 = self._format_position(matches[1].get("position"))
            return f"I can see {count} {plural}. The main ones are {p1} and {p2}."

    def generate(
        self,
        command: Optional[Dict[str, Any]],
        detections: Optional[List[Dict[str, Any]]],
        query_results: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a natural language text response for a given command, detections, and query memory results.

        Args:
            command (dict): Parsed command dict from ObjectCommandParser.
            detections (list): List of live detection dictionaries from perception pipeline.
            query_results (list): Structured match results from ObjectQueryMemory.find_object().

        Returns:
            str: Human-friendly text feedback.
        """
        if not isinstance(command, dict):
            return "I'm not sure what you're looking for. Try saying find my phone."

        intent        = command.get("intent")
        target_object = command.get("object")

        if intent != "find_object" or not target_object or not isinstance(target_object, str):
            return "I'm not sure what you're looking for. Try saying find my phone."

        clean_target = target_object.strip().lower()

        # --- Use ObjectQueryMemory results if available ---
        if query_results is not None:
            if not isinstance(query_results, list) or len(query_results) == 0:
                return f"I can't find your {clean_target} right now. It's not in my view."

            if len(query_results) == 1:
                match       = query_results[0]
                pos_phrase  = self._format_position(match.get("position"))
                dist_phrase = self._format_distance(match.get("distance"))
                is_current  = match.get("is_in_current_frame", False)
                frames_ago  = match.get("recency_frames_ago", 0)

                if is_current:
                    return self._build_response("found", clean_target, pos_phrase, dist_phrase)
                else:
                    # Recency-aware phrasing
                    if frames_ago <= 5:
                        recency = "just a second ago"
                    elif frames_ago <= 15:
                        recency = "a moment ago"
                    else:
                        recency = "a little while ago"

                    if dist_phrase:
                        return (
                            f"I last saw your {clean_target} {pos_phrase}, "
                            f"{dist_phrase}. That was {recency}."
                        )
                    else:
                        return f"I last saw your {clean_target} {pos_phrase}. That was {recency}."

            else:
                # Multiple instances
                count       = len(query_results)
                plural      = self._get_plural(clean_target)
                all_current = all(m.get("is_in_current_frame", False) for m in query_results)
                p1 = self._format_position(query_results[0].get("position"))
                p2 = self._format_position(query_results[1].get("position"))
                verb = "I can see" if all_current else "I recently saw"
                if count == 2:
                    return f"{verb} 2 {plural}. One is {p1} and the other is {p2}."
                else:
                    return f"{verb} {count} {plural}. The main ones are {p1} and {p2}."

        # --- Fallback: direct detections list ---
        if not isinstance(detections, list) or len(detections) == 0:
            return f"I can't find your {clean_target} right now. It's not in my view."

        matches = [
            d for d in detections
            if isinstance(d, dict)
            and str(d.get("class_name", "")).strip().lower() == clean_target
        ]

        if not matches:
            return f"I can't find your {clean_target} right now. It's not in my view."

        matches.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        if len(matches) == 1:
            return self._format_single_match(clean_target, matches[0])
        else:
            return self._format_multiple_matches(clean_target, matches)
