"""
generate_cure_report.py — Gera relatorio completo de descoberta de cura.

Uso:
    python scripts/generate_cure_report.py --disease Alzheimer
    python scripts/generate_cure_report.py --disease Parkinson --rounds 10
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.model_discovery_lab import ModelDiscoveryLab

REPORT_DIR = r"D:\novamind_data\cure_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate cure discovery report")
    parser.add_argument("--disease", default="Alzheimer",
                        help="Disease to discover cure for")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Max modeling rounds (default: dynamic)")
    args = parser.parse_args()

    print("=" * 65)
    print(f"CURE DISCOVERY REPORT GENERATOR")
    print(f"Disease: {args.disease}")
    print("=" * 65)

    # Discovery
    lab = ModelDiscoveryLab()
    start = time.time()
    proposal = lab.discover(args.disease, max_rounds=args.rounds)
    elapsed = time.time() - start

    # Report
    report = proposal.to_markdown()

    # Add performance info
    report += f"\n\n---\n"
    report += f"*Generated in {elapsed:.1f}s*\n"
    report += f"*Modeling rounds: {len(proposal.modeling_rounds)}*\n"
    report += f"*Models considered: {1 + len(proposal.rejected_models) + len(proposal.accepted_mutations)}*\n"

    # Save
    report_path = os.path.join(REPORT_DIR, f"cure_{args.disease.lower().replace(' ','_')}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved: {report_path}")
    print(f"Time: {elapsed:.1f}s")
    print(report[:2000])


if __name__ == "__main__":
    main()
