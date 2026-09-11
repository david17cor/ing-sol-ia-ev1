import json
from tools import consultar_estado_rut

def test_consultar_estado_rut_activo():
    res = consultar_estado_rut.invoke({"rut": "12345678-9"})
    data = json.loads(res)
    assert data["estado"] == "ACTIVO"
    assert data["nombre"] == "Ana María Silva"

def test_consultar_estado_rut_moroso():
    res = consultar_estado_rut.invoke({"rut": "11223344-5"})
    data = json.loads(res)
    assert data["estado"] == "DADO_DE_BAJA"
    assert data["meses_morosidad"] == 3

def test_consultar_estado_rut_no_encontrado():
    res = consultar_estado_rut.invoke({"rut": "00000000-0"})
    assert res == "RUT_NO_ENCONTRADO"