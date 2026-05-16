import pytest

from perceptrome.cli_main import build_parser
from perceptrome.genome.registry import DEFAULT_GENE_REGISTRY
from perceptrome.model_catalog import DNA_GENERATIVE_MODEL_TYPES, apply_training_model_overrides, normalize_model_type


def test_dna_model_catalog_exposes_diverse_architectures():
    expected = {
        "mlp",
        "transformer",
        "ssm",
        "conv",
        "recurrent",
        "wavenet",
        "mamba",
        "attention_pool",
        "bytenet",
        "tree",
        "hybrid",
        "hierarchical",
    }
    assert expected.issubset(set(DNA_GENERATIVE_MODEL_TYPES))
    assert normalize_model_type("cnn") == "conv"
    assert normalize_model_type("gru") == "recurrent"
    assert normalize_model_type("perceiver") == "attention_pool"
    assert normalize_model_type("s4") == "mamba"


def test_cli_model_type_accepts_expanded_dna_choices():
    parser = build_parser()

    stream_args = parser.parse_args(["stream", "--catalog", "config/plasmids_5.txt", "--model-type", "mamba"])
    generate_args = parser.parse_args(["generate-plasmid", "--model-type", "cnn"])

    assert stream_args.model_type == "mamba"
    assert generate_args.model_type == "cnn"


def test_genome_registry_model_type_choices_track_catalog():
    model_def = DEFAULT_GENE_REGISTRY.definitions["model_type"]
    assert model_def.choices == list(DNA_GENERATIVE_MODEL_TYPES)


def test_web_job_training_overrides_normalize_model_type():
    cfg = {"training": {"model_type": "mlp", "hidden_dim": 128}}

    out = apply_training_model_overrides(
        cfg,
        {
            "model_type": "cnn",
            "hidden_dim": 256,
            "transformer_layers": 6,
            "transformer_dropout": 0.2,
        },
    )

    assert out["training"]["model_type"] == "conv"
    assert out["training"]["hidden_dim"] == 256
    assert out["training"]["transformer_layers"] == 6
    assert out["training"]["transformer_dropout"] == 0.2


def test_web_job_training_overrides_reject_unknown_model_type():
    with pytest.raises(ValueError, match="Unsupported genomic DNA model_type"):
        apply_training_model_overrides({"training": {}}, {"model_type": "not_a_model"})
