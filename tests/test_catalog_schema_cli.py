import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace


if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = object
    np_stub.float32 = float
    np_stub.int64 = int
    np_stub.array = lambda v, dtype=None: list(v)
    np_stub.random = types.SimpleNamespace(Generator=object)
    sys.modules["numpy"] = np_stub

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")
    nn_stub.Module = object
    optim_stub = types.ModuleType("torch.optim")
    fn_stub = types.ModuleType("torch.nn.functional")
    utils_stub = types.ModuleType("torch.utils")
    data_stub = types.ModuleType("torch.utils.data")
    data_stub.DataLoader = object
    data_stub.TensorDataset = object

    torch_stub.nn = nn_stub
    torch_stub.optim = optim_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.optim"] = optim_stub
    sys.modules["torch.nn.functional"] = fn_stub
    sys.modules["torch.utils"] = utils_stub
    sys.modules["torch.utils.data"] = data_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = lambda *a, **k: None
    sys.modules["requests"] = requests_stub

from perceptrome.catalog_schema import parse_catalog_schema
from perceptrome.cli.commands import cmd_catalog_generate
from perceptrome.io_utils import read_catalog


class CatalogSchemaTests(unittest.TestCase):
    def test_parse_catalog_schema_json(self):
        with tempfile.TemporaryDirectory() as td:
            schema_path = os.path.join(td, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                f.write(
                    '{"schema_version":1,"seed":7,"categories":[{"name":"plasmid","count":2,"source":"acc.txt"}]}'
                )

            parsed = parse_catalog_schema(schema_path)
            self.assertEqual(parsed.schema_version, 1)
            self.assertEqual(parsed.seed, 7)
            self.assertEqual(parsed.categories[0].name, "plasmid")
            self.assertTrue(parsed.categories[0].source.endswith("acc.txt"))

    def test_cmd_catalog_generate_dedupes_across_categories(self):
        with tempfile.TemporaryDirectory() as td:
            c1 = os.path.join(td, "plasmid.txt")
            c2 = os.path.join(td, "virus.txt")
            with open(c1, "w", encoding="utf-8") as f:
                f.write("A1\nA2\nA3\n")
            with open(c2, "w", encoding="utf-8") as f:
                f.write("A2\nV1\nV2\n")

            schema_path = os.path.join(td, "schema.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                f.write(
                    """
                    {
                      "schema_version": 1,
                      "seed": 1,
                      "categories": [
                        {"name": "plasmid", "count": 2, "source": "plasmid.txt"},
                        {"name": "virus", "count": 2, "source": "virus.txt"}
                      ]
                    }
                    """
                )

            out = os.path.join(td, "catalog.txt")
            rc = cmd_catalog_generate(SimpleNamespace(schema=schema_path, output=out))
            self.assertEqual(rc, 0)

            accessions = read_catalog(out)
            self.assertEqual(accessions, ["A1", "A2", "V1", "V2"])


if __name__ == "__main__":
    unittest.main()
