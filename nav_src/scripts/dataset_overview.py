#!/usr/bin/env python3
"""Print structure and samples of NavGPT R2R dataset under ../datasets (from nav_src)."""

import json
import os
import sys
# Default: run from nav_src: python scripts/dataset_overview.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "R2R"))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ROOT
    if not os.path.isdir(root):
        print(f"目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("NavGPT R2R 数据根目录:", root)
    print("=" * 60)

    subdirs = [
        "annotations",
        "connectivity",
        "navigable",
        "observations_list_summarized",
        "observations_summarized",
        "objects_list",
    ]
    for name in subdirs:
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            print(f"\n[{name}] (缺失)")
            continue
        files = [f for f in os.listdir(p) if f.endswith(".json")]
        print(f"\n[{name}] 文件数: {len(files)}")
        if name == "annotations":
            for fn in sorted(files):
                fp = os.path.join(p, fn)
                with open(fp, encoding="utf-8") as f:
                    data = json.load(f)
                n = len(data) if isinstance(data, list) else len(data)
                keys = list(data[0].keys()) if isinstance(data, list) and data else []
                print(f"  - {fn}: {n} 条, 字段: {keys}")

    anno_dir = os.path.join(root, "annotations")
    sample_split = "R2R_val_unseen_instr.json"
    sample_path = os.path.join(anno_dir, sample_split)
    data = []
    if os.path.isfile(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            data = json.load(f)
        print("\n" + "=" * 60)
        print(f"标注样例（第一条）来自 {sample_split}:")
        print("=" * 60)
        print(json.dumps(data[0], indent=2, ensure_ascii=False))

        scans = [x.get("scan") for x in data if isinstance(x, dict)]
        print("\n统计: 该 split 中唯一 scan 数:", len(set(scans)))

    # Per-file structure for one scan
    scan = data[0]["scan"] if data else None
    vp = (data[0]["path"][0] if data[0].get("path") else None) if data else None

    if scan and vp:
        print("\n" + "=" * 60)
        print(f"同一 scan={scan} 下其它资源样例（viewpoint={vp[:16]}...）")
        print("=" * 60)

        sum_list = os.path.join(root, "observations_list_summarized", f"{scan}.json")
        if os.path.isfile(sum_list):
            with open(sum_list, encoding="utf-8") as f:
                obs = json.load(f)
            lines = obs.get(vp, [])[:2]
            print(f"\nobservations_list_summarized/{scan}.json")
            print(f"  该 scan 的 viewpoint 数: {len(obs)}")
            print("  某 viewpoint 前 2 条朝向描述:")
            for i, s in enumerate(lines):
                print(f"    [{i}] {s[:120]}..." if len(s) > 120 else f"    [{i}] {s}")

        sum_one = os.path.join(root, "observations_summarized", f"{scan}_summarized.json")
        if os.path.isfile(sum_one):
            with open(sum_one, encoding="utf-8") as f:
                sm = json.load(f)
            t = sm.get(vp, "")
            print(f"\nobservations_summarized/{scan}_summarized.json")
            print(f"  该 scan 条目数: {len(sm)}")
            print(f"  摘要示例: {t[:200]}..." if len(t) > 200 else f"  摘要示例: {t}")

        objp = os.path.join(root, "objects_list", f"{scan}.json")
        if os.path.isfile(objp):
            with open(objp, encoding="utf-8") as f:
                objs = json.load(f)
            per_vp = objs.get(vp)
            print(f"\nobjects_list/{scan}.json")
            print(f"  该 scan 的 viewpoint 数: {len(objs)}")
            if isinstance(per_vp, list):
                print(f"  每 viewpoint 为长度 {len(per_vp)} 的列表（对应 8 个水平朝向 × 上下？共 8 个扇区）")
                non_empty = sum(1 for x in per_vp if x)
                print(f"  该 viewpoint 非空扇区数: {non_empty}")
                for i, block in enumerate(per_vp):
                    if block:
                        print(f"    扇区[{i}] 物体示例: {list(block.keys())[:5]}")
                        break

        navp = os.path.join(root, "navigable", f"{scan}_navigable.json")
        if os.path.isfile(navp):
            with open(navp, encoding="utf-8") as f:
                nav = json.load(f)
            nbrs = nav.get(vp, {})
            print(f"\nnavigable/{scan}_navigable.json")
            print(f"  从当前 viewpoint 可直达的相邻点数: {len(nbrs)}")
            nid = next(iter(nbrs))
            print(f"  邻居 {nid[:16]}... 的字段: {list(nbrs[nid].keys())}")

        conp = os.path.join(root, "connectivity", f"{scan}_connectivity.json")
        if os.path.isfile(conp):
            with open(conp, encoding="utf-8") as f:
                conn = json.load(f)
            print(f"\nconnectivity/{scan}_connectivity.json")
            print(f"  该 scan 离散视点（节点）数: {len(conn)}")
            if conn:
                print(f"  单节点字段示例: {list(conn[0].keys())}")

    print("\n" + "=" * 60)
    print("说明（与代码对应）")
    print("=" * 60)
    print("""
- annotations: R2R 任务条目。instr 类文件每条含 scan、path（viewpoint id 序列）、
  instruction、instr_id、path_id、heading、distance；供 NavGPT 加载轨迹与指令。
- connectivity: Matterport 离散视点图，含 pose、与其它视点的可见性等，用于最短路等。
- navigable: 每个视点一步可到达的邻居及相对 heading/elevation/distance。
- observations_list_summarized: 每视点 8 个水平方向的场景英文描述列表（detail）。
- observations_summarized: 每视点整圈观察合并成一段摘要（给 history 用）。
- objects_list: 每视点 8 个扇区内的物体名及相对 heading/distance（供 LLM 工具链）。

完整 3D mesh 在 parser 里的 Matterport3D scan 目录（若已下载），本仓库主要用上述 JSON。
""")


if __name__ == "__main__":
    main()
