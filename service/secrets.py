from flask import abort

secrets_dict = {}


def put(secret):
    """Add a secret to the secret store

    :param secret: Dictionary of secret to be saved.
    :type secret: dict {"name": str, "value": str}
    """
    secret_name = secret.get("name")
    secret_value = secret.get("value")
    if secret_name not in secrets_dict:
        secrets_dict[secret_name] = secret_value
    return {"name": secret_name}


def get(name):
    """Add a secret to the secret store

    :param name: Name of the secret
    :type name: str
    :rtype: dict {"name": str, "value": str}
    """
    if name not in secrets_dict:
        abort(400, "Secret {} not found.".format(name))
    return {"name": name, "value": secrets_dict[name]}
