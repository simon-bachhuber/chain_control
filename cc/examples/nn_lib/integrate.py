def _explicit_euler(rhs, t, x, dt):
    dx = rhs(t, x)
    return x + dt * dx


def _runge_kutta(rhs, t, x, dt):
    h = dt
    k1 = rhs(t, x)
    k2 = rhs(t + h / 2, x + h * k1 / 2)
    k3 = rhs(t + h / 2, x + h * k2 / 2)
    k4 = rhs(t + h, x + h * k3)
    dx = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x + dt * dx


def integrate(rhs, x, t, dt, method):
    """
    Args:
        rhs: (t, x) -> dx
    """
    if method == "RK4":
        return _runge_kutta(rhs, t, x, dt)
    elif method == "EE":
        return _explicit_euler(rhs, t, x, dt)
    elif method == "no-integrate":
        return rhs(t, x)
    else:
        raise NotImplementedError()
