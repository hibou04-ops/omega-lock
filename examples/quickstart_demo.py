"""Compact offline replay of the installed package's declared score gates."""
from omega_lock import gate_scores


def main() -> None:
    cases = [
        ("location_shift", [1, 2, 3], [-100, -99, -98], False),
        ("normalized_collapse", [0.7, 0.8, 0.9], [0.01, 0.02, 0.03], False),
        ("unchanged_holdout", [1, 2, 3], [1, 2, 3], True),
    ]
    for name, train, holdout, expected in cases:
        result = gate_scores(train, holdout)
        assert result.deployment_authorized is expected, name
        prefix = "+ [PASS]" if result.deployment_authorized else "- [BLOCKED]"
        print(f"{prefix} {name}: deployment_authorized={result.deployment_authorized}")


if __name__ == "__main__":
    main()
