import numpy as np


def agente_reglas(e_k, dy_k, u_propuesto, fallo_sensor=False):
    """Agente basado en reglas explicables para asignar reservas (0% - 100%)."""
    if fallo_sensor:
        return 50.0, True, "Modo seguro: fallo o pérdida de medición"

    motivo = "Operación nominal PI"
    u_ajustado = u_propuesto
    alerta = False

    if e_k < -0.5 or dy_k < -0.15:
        u_ajustado = max(u_propuesto, 80.0)
        motivo = "Regla 2: Emergencia por caída de frecuencia"
    elif abs(e_k) < 0.03 and abs(dy_k) < 0.02:
        u_ajustado = 0.85 * u_propuesto
        motivo = "Regla 3: Banda de estabilidad, suavizado de reserva"
    elif e_k > 0.2:
        u_ajustado = min(u_propuesto, 15.0)
        motivo = "Regla 4: Sobre-frecuencia, reducción de reserva"

    u_final = float(np.clip(u_ajustado, 0.0, 100.0))
    return u_final, alerta, motivo


def simular_sistema(
    modo="PI",
    pasos=80,
    x0=-0.8,
    ref=0.0,
    a=0.92,
    b=0.015,
    kp=35.0,
    ki=6.0,
    u_fijo=25.0,
    usar_agente=True,
    perfil_d=None,
    ruido_std=0.0,
    prob_perdida=0.0,
    semilla=42,
):
    rng = np.random.default_rng(semilla)
    x = np.zeros(pasos + 1)
    y = np.zeros(pasos + 1)
    u = np.zeros(pasos)
    registro_alertas = []

    x[0] = x0
    y[0] = x[0] + (rng.normal(0.0, ruido_std) if ruido_std > 0 else 0.0)
    integral_e = 0.0
    ultima_y_valida = y[0]

    for k in range(pasos):
        d_k = perfil_d[k] if perfil_d is not None else 0.0
        fallo_sensor = False

        if k > 0 and prob_perdida > 0 and rng.random() < prob_perdida:
            fallo_sensor = True
            y_k = ultima_y_valida
        else:
            n_k = rng.normal(0.0, ruido_std) if (k > 0 and ruido_std > 0) else 0.0
            y_k = x[k] + n_k
            ultima_y_valida = y_k

        y[k] = y_k
        e_k = y[k] - ref
        dy_k = (y[k] - y[k - 1]) if k > 0 else 0.0

        if modo == "BASE":
            u_raw = u_fijo
        elif modo == "P":
            u_raw = -kp * e_k
        elif modo == "PI":
            integral_tentativa = np.clip(integral_e + e_k, -15.0, 15.0)
            u_raw = -kp * e_k - ki * integral_tentativa
            if 0.0 <= u_raw <= 100.0:
                integral_e = integral_tentativa
        else:
            raise ValueError("Modo inválido")

        u_raw = float(np.clip(u_raw, 0.0, 100.0))

        if usar_agente and modo != "BASE":
            u_k, alerta, motivo = agente_reglas(e_k, dy_k, u_raw, fallo_sensor=fallo_sensor)
            if alerta:
                registro_alertas.append((k, motivo))
        else:
            u_k = u_raw

        u[k] = u_k
        x[k + 1] = a * x[k] + b * u[k] + d_k

        if abs(x[k + 1]) > 2.5:
            registro_alertas.append((k + 1, "PARADA DE SEGURIDAD: desviación crítica"))
            x = x[: k + 2]
            u = u[: k + 1]
            break

    return x, u, registro_alertas


if __name__ == "__main__":
    x_pi, u_pi, alertas = simular_sistema(modo="PI", usar_agente=True, semilla=42)
    print(f"Verificación k=0 -> u[0]: {u_pi[0]:.1f}%, x[1]: {x_pi[1]:.4f}")
    print(f"Estado final x[80]: {x_pi[-1]:.4f}, Reserva final: {u_pi[-1]:.1f}%")
