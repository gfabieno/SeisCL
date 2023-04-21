from .fdcoefficient import FDCoefficients

def get_header_stencil(order, nd, local_memory=True, with_local_ops=False):

    if nd == 1:
        names = ["z"]
    elif nd == 2:
        names = ["x", "z"]
    elif nd == 3:
        names = ["x", "y", "z"]

    coefs = FDCoefficients(order=order, maxrerror=1)
    header = coefs.header()
    if local_memory:
        header += get_local_memory_loader(order, nd, with_ops=with_local_ops)
    for name in names:
        header += get_stencil(order, nd, name, forward=True,
                              local_memory=local_memory)
        header += get_stencil(order, nd, name, forward=False,
                              local_memory=local_memory)
    return header


def get_local_memory_loader(order, nd, with_ops=False):

    if nd == 1:
        names = ["z"]
    elif nd == 2:
        names = ["x", "z"]
    elif nd == 3:
        names = ["x", "y", "z"]

    header = "//Load in local memory with the halo\n"
    postr1 = ", ".join(["g.%s" % name for name in names])
    postr2 = postr1.replace("g.", "g.l")
    header += "#define load_local_in(v, lv) lv(%s)=v(%s)\n" % (postr2, postr1)
    header += "#define mul_local_in(v, lv) lv(%s)*=v(%s)\n" % (postr2, postr1)

    load_local_halox = """
    #define load_local_halox(v, lv) \\
    do{\\
        if (g.lx<2*FDOH)\\
            lv(g.lz, g.ly, g.lx-FDOH)=v(g.z, g.y, g.x-FDOH);\\
        if (g.lx+g.nlx-3*FDOH<FDOH)\\
            lv(g.lz, g.ly, g.nlx-3*FDOH)=v(g.z, g.y, g.nlx-3*FDOH);\\
        if (g.lx>(g.nlx-2*FDOH-1))\\
            lv(g.lz, g.ly, FDOH)=v(g.z, g.y, FDOH);\\
        if (g.lx-g.nlx+3*FDOH>(g.nlx-FDOH-1))\\
            lv(g.lz, g.ly, -g.nlx+3*FDOH)=v(g.z, g.y, -g.nlx+3*FDOH);\\
        } while(0)\n""".strip() + "\n"
    if nd == 2:
        load_local_halox = load_local_halox.replace("g.ly, ", "")
        load_local_halox = load_local_halox.replace("g.y, ", "")
    elif nd == 1:
        load_local_halox = ""
    header += load_local_halox
    if with_ops:
        header += load_local_halox.replace("load", "mul").replace("=", "*=")
        header += load_local_halox.replace("load", "add").replace("=", "+=")
        header += load_local_halox.replace("load", "sub").replace("=", "-=")
        header += load_local_halox.replace("load", "div").replace("=", "/=")

    load_local_haloy = """ 
    #define load_local_haloy(v, lv) \\
    do{\\
        if (g.ly<2*FDOH)\\
            lv(g.lz, g.ly-FDOH, g.lx)=v(g.z, g.y-FDOH, g.x);\\
        if (g.ly+g.nly-3*FDOH<FDOH)\\
            lv(g.lz, g.nly-3*FDOH, g.lx)=v(g.z, g.nly-3*FDOH, g.x);\\
        if (g.ly>(g.nly-2*FDOH-1))\\
            lv(g.lz, FDOH, g.lx)=v(g.z, FDOH, g.x);\\
        if (g.ly-g.nly+3*FDOH>(g.nly-FDOH-1))\\
            lv(g.lz, -g.nly+3*FDOH, g.lx)=v(g.z, -g.nly+3*FDOH, g.x);\\
    } while(0)\n""".strip()
    if nd < 3:
        load_local_haloy = ""
    header += load_local_haloy + "\n"
    if with_ops:
        header += load_local_haloy.replace("load", "mul").replace("=", "*=")
        header += load_local_haloy.replace("load", "add").replace("=", "+=")
        header += load_local_haloy.replace("load", "sub").replace("=", "-=")
        header += load_local_haloy.replace("load", "div").replace("=", "/=")

    load_local_haloz = """
    #define load_local_haloz(v, lv) \\
    do{\\
        if (g.lz<2*FDOH)\\
            lv(g.lz-FDOH, g.ly, g.lx)=v(g.z-FDOH, g.y, g.x);\\
        if (g.lz>(g.nlz-2*FDOH-1))\\
            lv(g.lz+FDOH, g.ly, g.lx)=v(g.z+FDOH, g.y, g.x);\\
    } while(0)\n""".strip()
    if nd == 1:
        load_local_haloz = load_local_haloz.replace(", g.ly, g.lx", "")
        load_local_haloz = load_local_haloz.replace(", g.y, g.x", "")
    elif nd == 2:
        load_local_haloz = load_local_haloz.replace("g.ly, ", "")
        load_local_haloz = load_local_haloz.replace("g.y, ", "")
    header += load_local_haloz + "\n"
    if with_ops:
        header += load_local_haloz.replace("load", "mul").replace("=", "*=")
        header += load_local_haloz.replace("load", "add").replace("=", "+=")
        header += load_local_haloz.replace("load", "sub").replace("=", "-=")
        header += load_local_haloz.replace("load", "div").replace("=", "/=")

    header = header.replace("FDOH", "%d" % (order//2))

    header += ""#define lvar(x, y, z) lvar["

    lvar = "#define lvar"
    if nd == 3:
        lvar += "(z, y, x) lvar[(x) * g.nlz * g.nly + (y) * g.nlz + (z)]\n"
    elif nd == 2:
        lvar += "(z, x) lvar[(x) * g.nlz + (z)]\n"
    elif nd == 1:
        lvar += "(x) lvar[(x)]\n"
    header += lvar

    return header


def get_stencil(order, nd, dim, forward=True, local_memory=True):

    if nd == 1:
        posp = "(g.z+%d)"
        posm = "(g.z-%d)"
    elif nd == 2:
        if dim == "x":
            posp = "(g.z, g.x+%d)"
            posm = "(g.z, g.x-%d)"
        elif dim == "z":
            posp = "(g.z+%d, g.x)"
            posm = "(g.z-%d, g.x)"
        else:
            raise ValueError("dim must be x or z for 2D")
    elif nd == 3:
        if dim == "x":
            posp = "(g.z, g.y, g.x+%d)"
            posm = "(g.z, g.y, g.x-%d)"
        elif dim == "y":
            posp = "(g.z, g.y+%d, g.x)"
            posm = "(g.z, g.y-%d, g.x)"
        elif dim == "z":
            posp = "(g.z+%d, g.y, g.x)"
            posm = "(g.z-%d, g.y, g.x)"
        else:
            raise ValueError("dim must be x, y or z for 3D")
    if local_memory:
        posp = posp.replace("g.", "g.l")
        posm = posm.replace("g.", "g.l")

    if forward:
        x = 1
        flag = "p"
    else:
        x = 0
        flag = "m"
    dxp = "#define D%s%s(v) (" % (dim, flag)
    dxp += ("+\\\n" + " "*len(dxp)).join(["HC%d*(v%s-v%s)" % (ii+1,
                                                              posp % (ii+x),
                                                              posm % (ii+1-x))
                                          for ii in range(order//2)])
    dxp += ")\n"

    return dxp


if __name__ == "__main__":
    import sys
    order = int(sys.argv[1])
    nd = int(sys.argv[2])
    dim = sys.argv[3]
    forward = bool(int(sys.argv[4]))
    local_memory = bool(int(sys.argv[5]))
    print(get_header_stencil(order, nd, dim, forward, local_memory))