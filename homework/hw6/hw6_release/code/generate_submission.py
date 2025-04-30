
import argparse
import inspect
from typing import Dict, List

from neural_networks import activations, layers, losses, models


def get_source_code_unstructured(unstructured_names: List[str]) -> Dict[str, str]:
    return {
        object_name: inspect.getsource(eval(object_name))
        for object_name in unstructured_names
    }


def get_source_code_structured(
    structured_names: Dict[str, List[str]]
) -> Dict[str, Dict[str, str]]:
    return {
        impl_type: get_source_code_unstructured(unstructured_names)
        for impl_type, unstructured_names in structured_names.items()
    }


def emit_markdown(source_code: Dict[str, str]) -> str:
    return "\n".join(
        f"""Implementation of `{object_name}`:

```python
{implementation}
```
"""
        for object_name, implementation in source_code.items()
    )


def emit_markdown_with_headings(
    source_code: Dict[str, Dict[str, str]], heading_level: int = 1
) -> str:
    return "\n".join(
        f"{heading_level * '#'} {heading}\n\n{emit_markdown(implementations)}\n"
        for heading, implementations in source_code.items()
    )


def emit_latex(source_code: Dict[str, str]) -> str:
    return "\n".join(
        f"""Implementation of \\texttt{{{object_name}}}:

\\begin{{lstlisting}}[language=Python]
{implementation}
\\end{{lstlisting}}
"""
        for object_name, implementation in source_code.items()
    )


def emit_latex_with_headings(
    source_code: Dict[str, Dict[str, str]], heading_level: int = 1
) -> str:
    return "\n".join(
        f"\\{(heading_level - 1) * 'sub'}section{{{heading}}}\n\n{emit_latex(implementations)}\n"
        for heading, implementations in source_code.items()
    )


# TODO: make sure that all student implementations are in this list
student_implementations = {
    "Activation Function Implementations:": [
        "activations.Linear",
        "activations.Sigmoid",
        "activations.ReLU",
        "activations.SoftMax",
    ],
    "Layer Implementations:": [
        "layers.FullyConnected",
        "layers.Pool2D",
        "layers.Conv2D.__init__",
        "layers.Conv2D._init_parameters",
        "layers.Conv2D.forward",
        "layers.Conv2D.backward",
    ],
    "Loss Function Implementations:": ["losses.CrossEntropy"],
    "Model Implementations:": [
        "models.NeuralNetwork.forward",
        "models.NeuralNetwork.backward",
        "models.NeuralNetwork.predict",
    ],
}


def flatten(structured_names: Dict[str, List[str]]) -> List[str]:
    return [name for lst in structured_names.values() for name in lst]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracts the relevant implementations out of the numerous Python files you edited, and emits Markdown or LaTeX."
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["markdown", "latex"],
        help="The output format, which can be either Markdown or LaTeX. Please specify in all lowercase.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="utf-8"),
        help="""Name of output file with appropriate extension.
    Please specify `foo.md` if you want to write to `foo.md`, instead of `foo`.""",
    )
    parser.add_argument(
        "--heading_level",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="""Level of headings.
    In case of LaTeX, 1 => \\section, 2 => \\subsection, 3 => \\subsubsection.
    In case of Markdown, 1 => #, 2 => ##, 3 => ###.
    """,
    )

    args = parser.parse_args()
    with args.output as f:
        if args.format == "latex":
            f.write(
                emit_latex_with_headings(
                    get_source_code_structured(student_implementations), args.heading_level
                )
            )
        elif args.format == "markdown":
            f.write(
                emit_markdown_with_headings(
                    get_source_code_structured(student_implementations), args.heading_level
                )
            )
        else:
            raise ValueError(f"Cannot produce output in format `{args.format}`")
