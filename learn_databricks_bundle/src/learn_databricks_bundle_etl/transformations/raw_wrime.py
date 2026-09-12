from pyspark import pipelines as dp  # type: ignore

PATH = "/Volumes/sandbox/shotaro_minami8e40a5/files/wrime-ver1.tsv"


@dp.table
def wrime():
    df = spark.read.option("header", "true").option("sep", "\t").csv(PATH)  # type: ignore # noqa: F821

    # Deltaで使用できない文字を置換
    for column in df.columns:
        new_column = (
            column.replace(".", "_")
            .replace(" ", "_")
            .replace(",", "_")
            .replace(";", "_")
            .replace("{", "_")
            .replace("}", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("\n", "_")
            .replace("\t", "_")
            .replace("=", "_")
        )

        df = df.withColumnRenamed(column, new_column)

    return df
