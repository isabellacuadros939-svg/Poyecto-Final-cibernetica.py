import numpy as np
import matplotlib.pyplot as plt


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


def calcular_metricas(x, u, ref=0.0):
    mae = float(np.mean(np.abs(x[1:] - ref)))
    error_final = float(np.abs(x[-1] - ref))
    uso_reserva = float(np.mean(u))
    sobreimpulso = float(np.max(np.maximum(0.0, x[1:] - ref)))
    return mae, error_final, uso_reserva, sobreimpulso


def ejecutar_experimentos_entrega3():
    pasos = 80
    semilla = 42

    d_nominal = np.zeros(pasos)
    d_mod = np.zeros(pasos)
    d_mod[20:50] = -0.04

    d_sev = np.zeros(pasos)
    d_sev[20:50] = -0.15

    escenarios = [
        ("A. Nominal", d_nominal, 0.0, 0.0),
        ("B. Perturbacion Moderada", d_mod, 0.0, 0.0),
        ("C. Perturbacion Severa", d_sev, 0.0, 0.0),
        ("D. Ruido e Info Incompleta", d_mod, 0.06, 0.15),
    ]

    print(f"{'Escenario':<28} | {'Estrategia':<12} | {'MAE':<8} | {'ErrFinal':<8} | {'Reserva(%)':<10} | {'Sobreimp.':<9}")
    print("-" * 90)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (nombre, perfil_d, ruido, perdida) in enumerate(escenarios):
        x_b, u_b, _ = simular_sistema(
            modo="BASE", pasos=pasos, usar_agente=False,
            perfil_d=perfil_d, ruido_std=ruido, prob_perdida=perdida, semilla=semilla
        )
        x_m, u_m, _ = simular_sistema(
            modo="PI", pasos=pasos, usar_agente=True,
            perfil_d=perfil_d, ruido_std=ruido, prob_perdida=perdida, semilla=semilla
        )

        mae_b, ef_b, res_b, sob_b = calcular_metricas(x_b, u_b)
        mae_m, ef_m, res_m, sob_m = calcular_metricas(x_m, u_m)

        print(f"{nombre:<28} | {'Linea Base':<12} | {mae_b:<8.4f} | {ef_b:<8.4f} | {res_b:<10.2f} | {sob_b:<9.4f}")
        print(f"{nombre:<28} | {'PI + Agente':<12} | {mae_m:<8.4f} | {ef_m:<8.4f} | {res_m:<10.2f} | {sob_m:<9.4f}")
        print("-" * 90)

        ax = axes[idx]
        ax.plot(x_b, label="Línea Base", linestyle="--")
        ax.plot(x_m, label="PI + Agente")
        ax.axhline(0.0, linestyle=":", color="black", label="Referencia")
        ax.set_title(nombre)
        ax.set_xlabel("Paso k")
        ax.set_ylabel("Desviación x[k]")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("escenarios_entrega3_comparacion.png", dpi=150)
    plt.close()

    print("\n--- PRUEBA DE SENSIBILIDAD (Variando Ki en Perturbación Moderada) ---")
    for ki_val in [1.0, 6.0, 15.0, 28.0]:
        x_s, u_s, _ = simular_sistema(
            modo="PI", pasos=pasos, ki=ki_val, usar_agente=True,
            perfil_d=d_mod, semilla=semilla
        )
        mae_s, ef_s, res_s, sob_s = calcular_metricas(x_s, u_s)
        print(f"Ki = {ki_val:<5.1f} | MAE: {mae_s:.4f} | ErrFinal: {ef_s:.4f} | ReservaProm: {res_s:.2f}% | Sobreimpulso: {sob_s:.4f}")


if __name__ == "__main__":
    ejecutar_experimentos_entrega3()
