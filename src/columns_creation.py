
if __name__ == '__main__':
    import json

    words = ([
         "age",
         "class of worker",
         "detailed industry recode",
         "detailed occupation recode",
         "education",
         "wage per hour",
         "enroll in edu inst last wk",
         "marital stat",
         "major industry code",
         "major occupation code",
         "race",
         "hispanic origin",
         "sex",
         "member of a labor union",
         "reason for unemployment",
         "full or part time employment stat",
         "capital gains",
         "capital losses",
         "dividends from stocks",
         "tax filer stat",
         "region of previous residence",
         "state of previous residence",
         "detailed household and family stat",
         "detailed household summary in household",
         "instance weight",
         "migration code-change in msa",
         "migration code-change in reg",
         "migration code-move within reg",
         "live in this house 1 year ago",
         "migration prev res in sunbelt",
         "num persons worked for employer",
         "family members under 18",
         "country of birth father",
         "country of birth mother",
         "country of birth self",
         "citizenship",
         "own business or self employed",
         "fill inc questionnaire for veteran's admin",
         "veterans benefits",
         "weeks worked in year",
         "year",
         "target"])

    cols = [word.replace(" ", "_") for word in words]
    my_dict = {key: value for key, value in zip(cols, range(0, len(cols)))}

    # Serialize the dictionary to a JSON string
    json_string = json.dumps(my_dict)

    with open("../json_files/data_cols.json", "w") as outfile:
        outfile.write(json_string)
