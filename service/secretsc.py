from flask import abort

secrets_dict = {}


def put(secret):
    secret_name = secret.get("name")
    secret_value = secret.get("value")
    if secret_name not in secrets_dict:
        secrets_dict[secret_name] = secret_value
    return {"name": secret_name}


def get(name):
    if name not in secrets_dict:
        abort(400, "Secret {} not found.".format(name))
    return {"name": name, "value": secrets_dict[name]}
