import csv

def save_analysis_results_to_csv(results: list[dict], output_path: str):
    """
    Save the analysis results (from Analyze.run()) to a CSV file.

    :param results: List of metric result dictionaries (from Analyze.run()).
    :param output_path: Path to the output CSV file.
    """
    csv_data = {}

    for entry in results:
        metric_name = entry['name']
        params = entry.get('parameters', {})
        value = entry.get('value', 'N/A')

        column_names = params.get('column_names')
        single_column = params.get('column_name')

        if column_names and column_names != "null":
            column_identifier = ', '.join(sorted(column_names))
        elif single_column and single_column != "null":
            column_identifier = single_column
        else:
            continue  # Skip if neither exists

        if column_identifier not in csv_data:
            csv_data[column_identifier] = {}

        # Construct metric descriptor
        other_params = [
            f"{k}={v}" for k, v in params.items()
            if k not in ('column_names', 'column_name')
        ]
        descriptor = f"{metric_name} ({'; '.join(other_params)})" if other_params else metric_name

        csv_data[column_identifier][descriptor] = 'N/A' if 'error' in entry else value

    # Prepare header
    headers = ['columns'] + sorted(
        list(set(desc for metrics in csv_data.values() for desc in metrics))
    )

    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for column_id, metrics in csv_data.items():
            row = {'columns': column_id}
            row.update(metrics)
            writer.writerow(row)

def save_evaluation_results_to_csv(results: list[dict], output_path: str):
    """
    Save the evaluation results (from Evaluate.run()) to a CSV file.

    :param results: List of evaluation result dicts.
    :param output_path: Path to output CSV file.
    """
    csv_data = {}

    for entry in results:
        metric_name = entry['name']
        data_pair = entry['data pair']
        params = entry.get('parameters', {})
        value = entry.get('value', 'N/A')
        error = entry.get('error')

        # Column key for the row
        categorical_columns = params.get('categorical_columns')
        if not categorical_columns:
            continue
        column_identifier = ', '.join(sorted(categorical_columns))

        measure_sum_columns = params.get('measure_sum_columns')
        edges = params.get('edges')

        if data_pair not in csv_data:
            csv_data[data_pair] = {}
        if column_identifier not in csv_data[data_pair]:
            csv_data[data_pair][column_identifier] = {}

        metric_descriptor = metric_name
        if measure_sum_columns:
            metric_descriptor += f"_{measure_sum_columns[0]}"

        metric_descriptor_list = []
        if edges:
            metric_descriptor_list.append(f'{metric_descriptor} (bin < {edges[0]})')
            for i in range(1, len(edges)):
                metric_descriptor_list.append(f'{metric_descriptor} (bin [{edges[i-1]}, {edges[i]}))')
            metric_descriptor_list.append(f'{metric_descriptor} (bin >= {edges[-1]})')

        if metric_descriptor_list:
            assert isinstance(value, dict)
            for i, descriptor in enumerate(metric_descriptor_list):
                csv_data[data_pair][column_identifier][descriptor] = 'N/A' if error else value.get(f'Bin {i}', 'N/A')
        else:
            csv_data[data_pair][column_identifier][metric_descriptor] = 'N/A' if error else value

    # Flatten and write CSV
    headers = ['data pair', 'columns'] + sorted(
        list(set(metric for pair_data in csv_data.values() for cols in pair_data.values() for metric in cols))
    )

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for data_pair, col_data in csv_data.items():
            for column_identifier, metrics in col_data.items():
                row = {
                    'data pair': data_pair,
                    'columns': column_identifier,
                    **metrics
                }
                writer.writerow(row)