from polpo.preprocessing.base import PreprocessingStep


class ManipulateDf(PreprocessingStep):
    def apply(self, data):
        # TODO: may need to do more

        hormones_df = data

        hormones_df["sessionID"] = [
            int(entry.split("-")[1]) for entry in hormones_df["sessionID"]
        ]
        hormones_df = hormones_df.drop(
            hormones_df[hormones_df["sessionID"] == 27].index
        )  # sess 27 is a repeat of sess 26
        # df = df[df["dayID"] != 27]  # sess 27 is a repeat of sess 26

        hormones_df.set_index(
            "sessionID", inplace=True, drop=False, verify_integrity=True
        )
        return hormones_df
