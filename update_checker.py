"""Consulta de versiones publicadas, sin Git ni modificación de archivos locales."""

import re

import requests


VERSION_URL = "https://raw.githubusercontent.com/palaciosluis3/IPP_Dashboard/main/version.json"
DOWNLOAD_URL = "https://github.com/palaciosluis3/IPP_Dashboard/archive/refs/heads/main.zip"


def parse_version(version):
    """Compara numéricamente versiones de tres componentes: 1.10.0 > 1.9.0."""
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("La versión debe tener el formato X.Y.Z (por ejemplo, 1.0.0).")
    return tuple(int(part) for part in version.split("."))


def validate_manifest(data):
    """Valida tanto el archivo local como la respuesta remota."""
    if not isinstance(data, dict):
        raise ValueError("version.json debe contener un objeto JSON.")
    parse_version(data.get("version"))
    if not isinstance(data.get("notes", ""), str):
        raise ValueError("El campo notes debe ser texto.")
    return {"version": data["version"], "notes": data.get("notes", "")}


def check_for_updates(local_version):
    """Devuelve available/current/ahead/error; los fallos de red no detienen la app."""
    local_number = parse_version(local_version)
    try:
        response = requests.get(
            VERSION_URL,
            timeout=(2, 3),
            headers={"Cache-Control": "no-cache"},
        )
        response.raise_for_status()
    except requests.RequestException:
        return {
            "status": "error",
            "message": "No se pudo consultar la versión publicada. Comprueba tu conexión o inténtalo más tarde.",
        }

    try:
        remote = validate_manifest(response.json())
    except ValueError:
        return {
            "status": "error",
            "message": "La información de actualización publicada no tiene un formato válido. Inténtalo más tarde.",
        }

    remote_number = parse_version(remote["version"])
    status = "available" if remote_number > local_number else "current"
    if remote_number < local_number:
        status = "ahead"
    return {"status": status, **remote}
