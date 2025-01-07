def run_experiments(experiments, repetitions, settings):
    """
    Run the experiments and return the results
    """

    overall_results = {}

    has_been_interrupted = False
    for i, experiment_class in enumerate(experiments):

        exp_name = f"{i}__{experiment_class.__name__}"

        try:
            results = []

            try:
                for rep in range(repetitions):
                    exp = experiment_class(seed=rep * 42, **settings)
                    exp.run()
                    results.append(exp.get_results())

            # catch keyboard interrupt to stop the experiments
            except KeyboardInterrupt:
                print("Keyboard interrupt detected. Exiting remaining repetitions.")

                # save the collected results
                overall_results[exp_name] = results

                # exit repetition loop or experiment loop
                if has_been_interrupted:
                    break
                has_been_interrupted = True

            overall_results[exp_name] = results

        # continue with other experiments if one fails
        except Exception as e:
            print(f"Error in experiment {exp_name}: {e}")

    return overall_results
