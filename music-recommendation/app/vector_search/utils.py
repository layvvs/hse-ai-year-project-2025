from typing import TypeAlias, Literal, Dict


TagName: TypeAlias = Literal['genre', 'instuments', 'vartags']
TagValues: TypeAlias = list[str]
Tags: TypeAlias = Dict[TagName, TagValues]
