from itertools import product
from context_aware_next_activity import evaluate_next_activity_with_context
import warnings

warnings.filterwarnings("ignore")


datasets = {
    "helpdesk": {
        "columns": [
            "Case ID",
            "Activity",
            "Complete Timestamp",
        ],
        "context_columns": [
            "Resource",
            # "workgroup",
            # "customer",
            # "seriousness",
            # "product",
            # "seriousness_2",
            # "service_level",
            # "service_type",
            # "support_section",
            # "responsible_section",
        ],
    },
    "sepsis": {
        "columns": [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        "context_columns": ["org:group"],
    },
    "bpi_challenge_2012": {
        "columns": [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        "context_columns": ["org:resource"],
    },
    "road_traffic_fine_management_process": {
        "columns": [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        "context_columns": [
            # "org:resource",
            # "amount",
            "dismissal",
            "vehicleClass",
            # "lifecycle:transition",
            # "article",
            # "points",
            # "expense",
            # "notificationType",
            # "lastSent",
            # "paymentAmount",
            # "matricola",
        ],
    },
    "bpi_challenge_2017": {
        "dataset_path": "BPI_Challenge_2017.csv",
        "columns": [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        "context_columns": [
            "org:resource",
            # "Action",
            # "EventOrigin",
            # "lifecycle:transition",
            # "case:LoanGoal",
            # "case:ApplicationType",
            # "case:RequestedAmount",
            # "FirstWithdrawalAmount",
            # "NumberOfTerms",
            # "MonthlyCost",
            # "CreditScore",
            # "OfferedAmount",
            # "OfferID",
        ],
    },
    "bpi_challenge_2020": {
        "dataset_path": "bpi_challenge_2020.csv",
        "columns": [
            "case:concept:name",
            "concept:name",
            "time:timestamp",
        ],
        "context_columns": [
            # "org:resource",
            # "org:role",
            # "case:Task",
            "case:OrganizationalEntity",
            "case:Project",
        ],
    },
}


def main():
    model_types = [
        "transformer-single",
        "transformer-multiple",
        "dense-single",
        "dense-multiple",
        "bi-lstm-single",
        "bi-lstm-multiple",
    ]
    prefix_length = ["Min", "One-Quarter", "Middle", "Max"]

    for name, doc in datasets.items():
        context_columns = doc["context_columns"]
        for config in product(model_types, prefix_length):
            print(f"Configs {config}")
            evaluate_next_activity_with_context(name, [], *config)
            for context_column in context_columns:
                print(f"Running dataset {name} with {context_column}")
                evaluate_next_activity_with_context(name, [context_column], *config)


if __name__ == "__main__":
    main()
