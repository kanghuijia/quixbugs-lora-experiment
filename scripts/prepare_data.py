import ast
import json
import random
import multiprocessing as mp
from pathlib import Path




ROOT = Path("/root/quixbugs_lora")

QUIXBUGS_DIR = ROOT / "QuixBugs"
BUGGY_DIR = QUIXBUGS_DIR / "python_programs"
CORRECT_DIR = QUIXBUGS_DIR / "correct_python_programs"
TEST_DIR = QUIXBUGS_DIR / "json_testcases"

OUTPUT_DIR = ROOT / "data" / "processed"
METADATA_DIR = ROOT / "data" / "metadata"

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"
METADATA_FILE = METADATA_DIR / "metadata.json"

TOTAL_SAMPLES = 50
VAL_SAMPLES = 5
TRAIN_SAMPLES = 45

RANDOM_SEED = 42


TEST_TIMEOUT = 5




def read_json_testcases(test_file):
    """
    读取 QuixBugs 的 JSONL 测试文件。

    例如：
        [[17, 0], 17]
        [[13, 13], 13]

    表示：
        gcd(17, 0) == 17
        gcd(13, 13) == 13
    """

    testcases = []

    with open(test_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"  [WARN] 无法解析 JSON: "
                    f"{test_file.name}, 第 {line_no} 行"
                )
                continue

            if not isinstance(item, list) or len(item) != 2:
                print(
                    f"  [WARN] 测试格式异常: "
                    f"{test_file.name}, 第 {line_no} 行"
                )
                continue

            args = item[0]
            expected = item[1]

            if not isinstance(args, list):
                print(
                    f"  [WARN] args 不是 list: "
                    f"{test_file.name}, 第 {line_no} 行"
                )
                continue

            testcases.append({
                "args": args,
                "expected": expected,
                "line_no": line_no
            })

    return testcases


def find_target_function(source_code, program_name):
    """
    找到 QuixBugs 程序中的目标函数。

    优先寻找：
        函数名 == 文件名

    如果找不到，则使用第一个顶层函数。
    """

    tree = ast.parse(source_code)

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not functions:
        return None

    
    for node in functions:
        if node.name == program_name:
            return node

    
    return functions[0]


def extract_function_code(source_code, program_name):
    """
    从完整 Python 文件中提取目标函数代码。
    """

    tree = ast.parse(source_code)

    target = find_target_function(source_code, program_name)

    if target is None:
        return None, None

    lines = source_code.splitlines()

    start = target.lineno - 1

    
    end = target.end_lineno

    function_code = "\n".join(lines[start:end])

    return target.name, function_code




def execute_function_worker(
    source_code,
    function_name,
    args,
    queue
):
    """
    在独立子进程中执行 Python 函数。

    这样即使 buggy 程序死循环，
    主进程也不会被永久卡住。
    """

    try:

        namespace = {
            "__name__": "__quixbugs__"
        }


        exec(source_code, namespace)

        if function_name not in namespace:
            queue.put({
                "status": "error",
                "error": f"找不到函数 {function_name}"
            })
            return

        func = namespace[function_name]

        result = func(*args)


        if hasattr(result, "__next__"):
            result = list(result)

        queue.put({
            "status": "success",
            "result": result
        })

    except Exception as e:

        queue.put({
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}"
        })


def execute_with_timeout(
    source_code,
    function_name,
    args,
    timeout=TEST_TIMEOUT
):
    """
    带超时执行。

    返回：

        status = success
        status = error
        status = timeout
    """

    ctx = mp.get_context("fork")

    queue = ctx.Queue()

    process = ctx.Process(
        target=execute_function_worker,
        args=(
            source_code,
            function_name,
            args,
            queue
        )
    )

    process.start()

    process.join(timeout)


    if process.is_alive():

        process.terminate()
        process.join()

        return {
            "status": "timeout",
            "result": None,
            "error": f"执行超过 {timeout} 秒"
        }


    if queue.empty():

        return {
            "status": "error",
            "result": None,
            "error": "子进程没有返回结果"
        }

    return queue.get()




def result_matches(result, expected):
    """
    判断程序实际输出是否等于 testcase 的 expected。
    """

    return result == expected




def format_test(function_name, args, expected):
    """
    转换成类似：

        assert gcd(*[17, 0]) == 17
    """

    return (
        f"assert {function_name}(*{repr(args)}) "
        f"== {repr(expected)}"
    )




def build_samples():

    print("=" * 70)
    print("QuixBugs data preparation")
    print("=" * 70)

    print(f"QuixBugs directory:")
    print(QUIXBUGS_DIR)



    if not QUIXBUGS_DIR.exists():
        raise FileNotFoundError(
            f"QuixBugs 不存在：{QUIXBUGS_DIR}"
        )

    if not BUGGY_DIR.exists():
        raise FileNotFoundError(
            f"buggy 程序目录不存在：{BUGGY_DIR}"
        )

    if not CORRECT_DIR.exists():
        raise FileNotFoundError(
            f"correct 程序目录不存在：{CORRECT_DIR}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"测试目录不存在：{TEST_DIR}"
        )

    buggy_files = sorted(BUGGY_DIR.glob("*.py"))

    print(
        f"发现 buggy Python 文件: "
        f"{len(buggy_files)}"
    )



    candidates = []

    skipped_no_correct = 0
    skipped_no_test = 0
    skipped_parse = 0
    skipped_execution = 0
    skipped_bad_oracle = 0

    total_tests = 0
    total_timeouts = 0



    for index, buggy_file in enumerate(buggy_files, start=1):

        program_name = buggy_file.stem

        print(
            f"\n[{index}/{len(buggy_files)}] "
            f"处理: {program_name}"
        )

        correct_file = (
            CORRECT_DIR / buggy_file.name
        )

        test_file = (
            TEST_DIR / f"{program_name}.json"
        )



        if not correct_file.exists():

            print(
                "  [SKIP] 没有对应的 correct Python 文件"
            )

            skipped_no_correct += 1
            continue



        if not test_file.exists():

            print(
                "  [SKIP] 没有对应的 JSON testcase"
            )

            skipped_no_test += 1
            continue



        try:

            buggy_source = buggy_file.read_text(
                encoding="utf-8"
            )

            correct_source = correct_file.read_text(
                encoding="utf-8"
            )

            buggy_function_name, buggy_function = (
                extract_function_code(
                    buggy_source,
                    program_name
                )
            )

            correct_function_name, correct_function = (
                extract_function_code(
                    correct_source,
                    program_name
                )
            )

        except Exception as e:

            print(
                f"  [SKIP] Python AST 解析失败: {e}"
            )

            skipped_parse += 1
            continue

        if buggy_function is None:

            print(
                "  [SKIP] 找不到 buggy function"
            )

            skipped_parse += 1
            continue

        if correct_function is None:

            print(
                "  [SKIP] 找不到 correct function"
            )

            skipped_parse += 1
            continue

        function_name = buggy_function_name



        testcases = read_json_testcases(test_file)

        print(
            f"  testcase 数量: {len(testcases)}"
        )

        if not testcases:
            print("  [SKIP] 没有有效 testcase")
            continue



        program_failures = 0

        for test_index, testcase in enumerate(
            testcases
        ):

            args = testcase["args"]
            expected = testcase["expected"]
            line_no = testcase["line_no"]

            total_tests += 1



            correct_result = execute_with_timeout(
                correct_source,
                correct_function_name,
                args
            )


            if correct_result["status"] == "timeout":

                total_timeouts += 1

                print(
                    f"    [SKIP] testcase "
                    f"{test_index + 1}: "
                    f"correct 程序也超时"
                )

                skipped_execution += 1
                continue


            if correct_result["status"] == "error":

                print(
                    f"    [SKIP] testcase "
                    f"{test_index + 1}: "
                    f"correct 程序执行失败"
                )

                skipped_execution += 1
                continue



            if not result_matches(
                correct_result["result"],
                expected
            ):

                print(
                    f"    [SKIP] testcase "
                    f"{test_index + 1}: "
                    f"expected 与 correct 程序结果不一致"
                )

                skipped_bad_oracle += 1
                continue



            buggy_result = execute_with_timeout(
                buggy_source,
                buggy_function_name,
                args
            )



            if buggy_result["status"] == "timeout":

                total_timeouts += 1

                print(
                    f"    [FOUND] testcase "
                    f"{test_index + 1}: "
                    f"buggy 程序超时"
                )

                is_failure = True



            elif buggy_result["status"] == "error":

                print(
                    f"    [FOUND] testcase "
                    f"{test_index + 1}: "
                    f"buggy 程序报错"
                )

                is_failure = True



            else:

                is_failure = not result_matches(
                    buggy_result["result"],
                    expected
                )

                if is_failure:

                    print(
                        f"    [FOUND] testcase "
                        f"{test_index + 1}: "
                        f"buggy 输出错误"
                    )



            if is_failure:

                failing_test = format_test(
                    function_name,
                    args,
                    expected
                )

                sample = {
                    "instruction": "修复下面代码使测试通过",

                    "input": (
                        buggy_function.strip()
                        + "\n\n"
                        + failing_test
                    ),

                    "output": correct_function.strip()
                }

                metadata = {
                    "program": program_name,
                    "buggy_file": str(
                        buggy_file.relative_to(ROOT)
                    ),
                    "correct_file": str(
                        correct_file.relative_to(ROOT)
                    ),
                    "test_file": str(
                        test_file.relative_to(ROOT)
                    ),
                    "test_index": test_index,
                    "test_line": line_no,
                    "args": args,
                    "expected": expected,
                    "function_name": function_name,
                    "buggy_result_status": buggy_result["status"],
                    "buggy_error": buggy_result.get("error")
                }

                candidates.append({
                    "sample": sample,
                    "metadata": metadata
                })

                program_failures += 1

        print(
            f"  本程序找到失败 testcase: "
            f"{program_failures}"
        )



    print("\n" + "=" * 70)
    print("数据构造完成")
    print("=" * 70)

    print(
        f"总 testcase 数量: {total_tests}"
    )

    print(
        f"找到的 buggy failing testcase: "
        f"{len(candidates)}"
    )

    print(
        f"超时 testcase: "
        f"{total_timeouts}"
    )

    print(
        f"跳过：没有 correct 文件: "
        f"{skipped_no_correct}"
    )

    print(
        f"跳过：没有 testcase: "
        f"{skipped_no_test}"
    )

    print(
        f"跳过：AST 解析失败: "
        f"{skipped_parse}"
    )

    print(
        f"跳过：correct 执行失败/超时: "
        f"{skipped_execution}"
    )

    print(
        f"跳过：expected 与 correct 不一致: "
        f"{skipped_bad_oracle}"
    )



    if len(candidates) < TOTAL_SAMPLES:

        print("\n[ERROR]")
        print(
            f"只有 {len(candidates)} 个有效修复样本，"
            f"不足要求的 {TOTAL_SAMPLES} 个。"
        )

        print(
            "\n请不要继续进行 LoRA 微调。"
        )

        print(
            "需要先解决数据数量问题。"
        )

        return



    random.seed(RANDOM_SEED)

    random.shuffle(candidates)

    selected = candidates[:TOTAL_SAMPLES]

    val_data = selected[:VAL_SAMPLES]
    train_data = selected[VAL_SAMPLES:]



    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



    with open(
        TRAIN_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for item in train_data:

            f.write(
                json.dumps(
                    item["sample"],
                    ensure_ascii=False
                )
                + "\n"
            )

    with open(
        VAL_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for item in val_data:

            f.write(
                json.dumps(
                    item["sample"],
                    ensure_ascii=False
                )
                + "\n"
            )

 

    metadata = {

        "config": {
            "total_samples": TOTAL_SAMPLES,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "random_seed": RANDOM_SEED,
            "test_timeout": TEST_TIMEOUT
        },

        "selected_samples": [
            item["metadata"]
            for item in selected
        ]
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

 

    print("\n" + "=" * 70)
    print("最终数据集")
    print("=" * 70)

    print(
        f"训练集: {TRAIN_FILE}"
    )

    print(
        f"验证集: {VAL_FILE}"
    )

    print(
        f"Metadata: {METADATA_FILE}"
    )

    print()

    print(
        f"训练样本数量: {len(train_data)}"
    )

    print(
        f"验证样本数量: {len(val_data)}"
    )

    print(
        f"总样本数量: "
        f"{len(train_data) + len(val_data)}"
    )

    print("\n数据准备成功！")




if __name__ == "__main__":

    build_samples()
