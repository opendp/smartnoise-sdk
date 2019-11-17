import mlflow
import json
import sys

from burdock.client import get_dataset_client
from burdock.data.adapters import load_dataset, load_metadata
from burdock.query.sql import QueryParser, Validate, Rewriter, MetadataLoader
from pandasql import sqldf


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    budget = float(sys.argv[2])
    query = sys.argv[3]

    with mlflow.start_run():
        dataset_document = get_dataset_client().read(dataset_name, budget)
        df_for_diffpriv1234 = load_dataset(dataset_document)
        df_name = "df_for_diffpriv1234"
        metadata = load_metadata(dataset_document)
        q = QueryParser(metadata).query(query)
        try:
            Validate().validateQuery(q, metadata)
        except Exception as e:
            raise Exception("Failed in validating query: {} eith exception {}".format(q, e))


        q_rewritten = Rewriter(metadata).query(q)
        clean_q = lambda q: str(q).replace(".".join(2 * [dataset_name]), df_name)

        q_result = sqldf(clean_q(q), locals())

        q_rewritten_result = sqldf(clean_q(q_rewritten), locals())
        result = {"original_query": str(q),
                  "final_query": str(q_rewritten),
                  "query_result": q_result.to_dict(),
                  "final_query_result": q_rewritten_result.to_dict()}

        with open("result.json", "w") as stream:
            json.dump(result, stream)
        mlflow.log_artifact("result.json")
