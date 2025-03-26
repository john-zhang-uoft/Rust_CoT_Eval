
from cdifflib import CSequenceMatcher
from camel_converter import to_snake
from typing import List


class ParseError(Exception):
    pass

class ContentParser:

    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.

        # Iterate through lines and remove all comments
        lines = content.splitlines()
        cleaned_lines = []
        for line in lines:
            if "//" in line:
                line = line.split("//")[0]
            cleaned_lines.append(line)

        if "```" in content:
            content = content.split("```")[1]
        # first parse with assumption that content has description
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag == "insert":
            return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                content = content.split(entry_point)[-1]
                return "".join(content.splitlines(keepends=True)[1:])
        raise ParseError(f"Prompt is not in content:\n{content}")
    @staticmethod
    def _entry_point_variations(entry_point: str) -> List[str]:
        # NOTE: workaround dataset's bug with entry point naming
        return [
            entry_point,
            to_snake(entry_point),
            entry_point[0].lower() + entry_point[1:],
        ]

    def __call__(self, prompt: str, content: str, entry_point: str):
        # NOTE: Model doesn't follow instructions directly:
        # adds description of change and sometimes fixes
        # typos, or other "bugs" in description.

        # Remove Rust-style comments which can include bad characters or code
        if "///" in content or "//" in content:
            # Process line by line to handle both doc comments (///) and regular comments (//)
            lines = content.splitlines()
            cleaned_lines = []
            for line in lines:
                # Remove everything after // or /// on each line
                if "//" in line:
                    line = line.split("//")[0]
                cleaned_lines.append(line)
            content = "\n".join(cleaned_lines)

        if "```" in content:
            content = content.split("```rust")[1].split("```")[0]

        # if main function is after the entry point, remove it
        # Check for main function in different formats
        main_patterns = [
            "fn main() {",
            "fn main(){",
            "fn main() {\n",
            "fn main(){\n"
        ]
        
        for pattern in main_patterns:
            if pattern in content and entry_point in content:
                # If main appears after the entry point, remove it and everything after
                entry_pos = content.find(entry_point)
                main_pos = content.find(pattern)
                if main_pos > entry_pos:
                    content = content[:main_pos]

        # first parse with assumption that content has description
        matcher = CSequenceMatcher(None, prompt, content)
        tag, _, _, j1, j2 = matcher.get_opcodes()[-1]
        if tag == "insert":
            return content[j1:j2]
        # second parse content with assumption that model wrote code without description
        for entry_point in self._entry_point_variations(entry_point):
            if entry_point in content:
                content = content.split(entry_point)[-1]
                return "".join(content.splitlines(keepends=True)[1:])
        raise ParseError(f"Prompt is not in content:\n{content}")
