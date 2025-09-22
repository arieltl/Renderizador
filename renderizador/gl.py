#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    transform_stack = []
    viewpoint_transform = np.identity(4)
    camera_transform = np.identity(4)
    camera_perspective_transform = np.identity(4)
    screen_transform = np.identity(4)
    perspective_transform = np.identity(4)
    supersample = 1

    directionalLightEnabled = False
    directionalLight_ambientIntensity = 0.0
    directionalLight_color = [1, 1, 1]
    directionalLight_intensity = 1.0
    directionalLight_direction = [0, 0, 1]


    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

       

        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]
        for i in range(0, len(point), 2):
            pos_x = int(point[i] * GL.supersample)
            pos_y = int(point[i + 1] * GL.supersample)


            gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, color)  # altera pixel (u, v, tipo, r, g, b)

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]
        def drawSegment(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            error = dx + dy

            while True:
                if x0 > 0 and y0 > 0 and x0 < GL.width and y0 < GL.height:
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, color)
                e2 = 2 * error
                if e2 >= dy:
                    if x0 == x1: break
                    error = error + dy
                    x0 = x0 + sx
                if e2 <= dx:
                    if y0 == y1: break
                    error = error + dx
                    y0 = y0 + sy


        for i in range(0, len(lineSegments)-2, 2):

            x0 = int(lineSegments[i] * GL.supersample) 
            y0 = int(lineSegments[i + 1] * GL.supersample)
            x1 = int(lineSegments[i + 2] * GL.supersample)
            y1 = int(lineSegments[i + 3] * GL.supersample)
            drawSegment(x0, y0, x1, y1)

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).


        def drawPoint(x,y,color):
            if (0 <= x < GL.width and
                0 <= y < GL.height):
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)
        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]
        x = 0
        y = int(radius * GL.supersample)
        d = 1 - int(radius * GL.supersample)
        while x <= y:

            drawPoint(x, y, color)
            drawPoint(-x, y, color)
            drawPoint(x, -y, color)
            drawPoint(-x, -y, color)
            drawPoint(y, x, color)
            drawPoint(-y, x, color)
            drawPoint(y, -x, color)
            drawPoint(-y, -x, color)
            if d < 0:
                d += 2 * x + 3
            else:
                d += 2 * (x - y) + 5
                y -= 1
            x += 1

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).



        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]
        for i in range(0, len(vertices), 6):
            a = np.array([vertices[i] * GL.supersample, vertices[i+1] * GL.supersample])
            b = np.array([vertices[i+2] * GL.supersample, vertices[i+3] * GL.supersample])
            c = np.array([vertices[i+4] * GL.supersample, vertices[i+5] * GL.supersample])


            def L(P0,P1,P):
                a = P - P0
                b = P1 - P0
                m = [[a[0],a[1]],[b[0],b[1]]]
                return np.linalg.det(m)

            def inside(x,y):
                return (L(a,b,[x,y]) > 0) and (L(b,c,[x,y]) > 0) and (L(c,a,[x,y]) > 0)

            min_x = max(0, int(min(a[0], b[0], c[0])))
            min_y = max(0, int(min(a[1], b[1], c[1])))
            max_x = min(GL.width, int(max(a[0], b[0], c[0])))
            max_y = min(GL.height, int(max(a[1], b[1], c[1])))
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    if inside(x,y):
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)
    @staticmethod
    def draw_triangle(a, b, c, transform, color):
        """Draw a single triangle with given vertices, transform matrix, and color."""
        # Convert to homogeneous coordinates
        a = np.array([a[0], a[1], a[2], 1.0])
        b = np.array([b[0], b[1], b[2], 1.0])
        c = np.array([c[0], c[1], c[2], 1.0])
        
        # Apply transform
        a = transform @ a
        b = transform @ b
        c = transform @ c
        
        # Normalize
        a = a / a[3]
        b = b / b[3]
        c = c / c[3]
        
        # Take screen coordinates
        a = a[:2]
        b = b[:2]
        c = c[:2]
        
        def L(P0, P1, P):
            a = P - P0
            b = P1 - P0
            m = [[a[0], a[1]], [b[0], b[1]]]
            return np.linalg.det(m)

        def inside(x, y):
            return (L(a, b, [x, y]) > 0) and (L(b, c, [x, y]) > 0) and (L(c, a, [x, y]) > 0)

        min_x = max(0, int(min(a[0], b[0], c[0])))
        min_y = max(0, int(min(a[1], b[1], c[1])))
        max_x = min(GL.width, int(max(a[0], b[0], c[0]))+1)
        max_y = min(GL.height, int(max(a[1], b[1], c[1]))+1)
        
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if inside(x, y):
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)


    @staticmethod
    def draw_triangle_2d(a, b, c, color=None, colors=None, zs=None, uvs=None, texture=None, transparency=None,material = None,normal = None, camera_space_coords= None):

        #
        za = a[2]
        zb = b[2]
        zc = c[2]


        a = a[:2]
        b = b[:2]
        c = c[:2]



        def L(P0, P1, P):
            a = P - P0
            b = P1 - P0
            m = [[a[0], a[1]], [b[0], b[1]]]
            return np.linalg.det(m)

        def inside(x, y):
            return (L(a, b, [x, y]) >=  0) and (L(b, c, [x, y]) >= 0) and (L(c, a, [x, y]) >= 0)

        min_x = max(0, int(min(a[0], b[0], c[0])))
        min_y = max(0, int(min(a[1], b[1], c[1])))
        max_x = min(GL.width, int(max(a[0], b[0], c[0]))+1)
        max_y = min(GL.height, int(max(a[1], b[1], c[1]))+1)

        # Precompute oriented triangle area (twice the area). Requires CCW winding for inside()
        area2 = L(a, b, c)
 
        
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                if inside(x, y):
                
                    # Barycentric weights
                    alpha = L([x, y], b, c) / area2
                    beta  = L([x, y], c, a) / area2
                    gamma = L([x, y], a, b) / area2

                    out_color = color
                    ndc_z = (alpha * za + beta * zb + gamma * zc)
                    buffer_z = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)[0]
                    if ndc_z >= buffer_z:
                        continue
                    gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [ndc_z])
                    z = None
                    if zs is not None:
                        z0, z1, z2 = zs
                        denom_z = (alpha / z0 + beta / z1 + gamma / z2)
                        z = 1.0 / denom_z


                    if colors is not None and colors[0] is not None:
                        if z is not None:
                            w0 = (alpha / z0)
                            w1 = (beta / z1)
                            w2 = (gamma / z2)
                            rgb = w0 * colors[0] + w1 * colors[1] + w2 * colors[2]
                            out_color = [int(vv * 255 * z) for vv in rgb]

                        else:
                            rgb = alpha * colors[0] + beta * colors[1] + gamma * colors[2]
                            out_color = [int(vv * 255) for vv in rgb]


                    # Texturing (requires uvs, texture, and zs for perspective-correct UV)
                    if texture is not None and uvs is not None and z is not None:
                        (u0, v0), (u1, v1), (u2, v2) = uvs
                        u = (alpha * (u0 / z0) + beta * (u1 / z1) + gamma * (u2 / z2)) * z
                        v = (alpha * (v0 / z0) + beta * (v1 / z1) + gamma * (v2 / z2)) * z

                        w_tex, h = texture.shape[0], texture.shape[1]
                        tx = int(u * w_tex)
                        ty = int((1-v) * h)
                        if tx < 0: tx = 0
                        if ty < 0: ty = 0
                        if tx >= w_tex: tx = w_tex - 1
                        if ty >= h: ty = h - 1
                        texel = texture[tx, ty]
                        tr = int(texel[0])
                        tg = int(texel[1])
                        tb = int(texel[2])

                        out_color = [tr, tg, tb]
                    if transparency is not None and transparency != 1.0 and material is None:
                        old_color = [float(v) for v in gpu.GPU.read_pixel([x, y], gpu.GPU.RGB8)]
                       
                        new_color = [float(v) for v in out_color]
                        out_color = [int(ov *transparency + nv * (1 - transparency)) for ov, nv in zip(old_color, new_color)]
                    
                    if normal is not None and material is not None and (GL.headlight or GL.directionalLightEnabled):
                        # Light parameters - use headlight or directional light
                        if GL.headlight:
                            light_direction = np.array([0, 0, 1])  # Headlight direction (towards viewer)
                            Ii = 1  # Light intensity
                            Iia = 0  # Ambient intensity
                            ILrgb = [1, 1, 1]  # Light color
                        else:
                            light_direction = np.array(GL.directionalLight_direction)
                            Ii = GL.directionalLight_intensity 
                            Iia = GL.directionalLight_ambientIntensity
                            ILrgb = GL.directionalLight_color
                        
                        OErgb = np.array(material.get("emissiveColor", [0.0, 0.0, 0.0]), dtype=float)
                        
                        # Material properties
                        diffuseColor = np.array(material.get("diffuseColor", [1, 1, 1]))
                        specularColor = np.array(material.get("specularColor", [1, 1, 1]))
                        shininess = material.get("shininess", 0.2) * 128  # X3D shininess is 0-1, convert to 0-128
                        
                        # Calculate current point in camera space using barycentric interpolation
                        if camera_space_coords is not None:
                            camera_space_a, camera_space_b, camera_space_c = camera_space_coords
                            current_point_camera = (alpha * camera_space_a + 
                                                   beta * camera_space_b + 
                                                   gamma * camera_space_c)
                        else:
                            # Fallback if camera space coords not available
                            current_point_camera = np.array([0, 0, -1])
                        
                        # Viewer vector V: from point to camera (camera is at origin in camera space)
                        V = -current_point_camera[:3]  # Vector from point to camera origin
                        V_norm = np.linalg.norm(V)
                        if V_norm > 0:
                            V = V / V_norm  # Normalize
                      
                        
                        # Light vector L_vec (normalized)
                        L_vec = light_direction / np.linalg.norm(light_direction)
                        
                        # Normal vector N (already normalized)
                        N = normal
                        
                        # Diffuse component: Ii × ODrgb × (N · L_vec)
                        NdotL = max(0, np.dot(N, L_vec))  # Clamp to positive
                        # print(f"normal: {N}")
                        # print(f"NdotL: {NdotL}")
                        diffuse_i = Ii * diffuseColor * NdotL
                        
                        # Specular component: Ii × OSrgb × (N · H)^shininess
                        # where H = (L_vec + V) / |L_vec + V| (half vector)
                        H = L_vec + V
                        H_norm = np.linalg.norm(H)
                        if H_norm > 0:
                            H = H / H_norm  # Normalize half vector
                            NdotH = max(0, np.dot(N, H))  # Clamp to positive
                            specular_i = Ii * specularColor * (NdotH ** shininess)
                        else:
                            specular_i = np.array([0, 0, 0])
                        
                        ambient_i = Iia * np.array(ILrgb) * diffuseColor
                        
                        # Combine emissive, ambient, diffuse and specular
                        final_color = OErgb + ambient_i + diffuse_i + specular_i
                        out_color = [min(255, max(0, int(v * 255))) for v in final_color]

                    gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, out_color)


    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]


        print("set")
        # Pre-calculate the composed transform matrix
        composed_transform = GL.viewpoint_transform @ GL.transform_stack[-1]
        difuseColor = colors.get("diffuseColor", [1, 1, 1])

        
        for i in range(0, len(point), 9):
            # Extract triangle vertices
            a = np.array([point[i], point[i+1], point[i+2],1])
            b = np.array([point[i+3], point[i+4], point[i+5],1])
            c = np.array([point[i+6], point[i+7], point[i+8],1])

            if GL.headlight:
                a_world = GL.transform_stack[-1] @ a
                b_world = GL.transform_stack[-1] @ b
                c_world = GL.transform_stack[-1] @ c

                camera_space_a = GL.camera_transform @ a_world
                camera_space_b = GL.camera_transform @ b_world
                camera_space_c = GL.camera_transform @ c_world
                camera_space_a = camera_space_a[:3] / camera_space_a[3]
                camera_space_b = camera_space_b[:3] / camera_space_b[3]
                camera_space_c = camera_space_c[:3] / camera_space_c[3]
                normal = np.cross(camera_space_b[:3] - camera_space_a[:3], camera_space_c[:3] - camera_space_a[:3])
                
                normal = normal / np.linalg.norm(normal)




                color = difuseColor
        
            #precompute the transform
            a = composed_transform @ a
            b = composed_transform @ b
            c = composed_transform @ c
            a = a[:3] / a[3]
            b = b[:3] / b[3]
            c = c[:3] / c[3]
            if GL.headlight:
                GL.draw_triangle_2d(a, b, c, normal=normal, material=colors, camera_space_coords= [camera_space_a, camera_space_b, camera_space_c] )
            else:
                GL.draw_triangle_2d(a, b, c, color=color, transparency=colors.get("transparency", None))

            


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        global viewpoint_transform

        eye = np.array(position)
        rotation = GL.rotation_mat(orientation[:3], orientation[3])[:3, :3]
        up = np.matmul(rotation, np.array([0, 1, 0]))
    
        forward = np.matmul(rotation, np.array([0, 0, -1]))
        GL.forward = forward
        at = eye + forward
        # divide by 4th component

        w = (at - eye)[:3]
        up = up[:3]
        w = w / float(np.linalg.norm(w))

        u = np.linalg.cross(w, up)
        u = u / float(np.linalg.norm(u))

        v = np.linalg.cross(u, w)
        v = v / float(np.linalg.norm(v))

        R = np.array([
            [u[0], v[0], -w[0], 0],
            [u[1], v[1], -w[1], 0],
            [u[2], v[2], -w[2], 0],
            [0, 0, 0, 1]
        ]).T

        E = np.array([
            [1, 0, 0, -eye[0]],
            [0, 1, 0, -eye[1]],
            [0, 0, 1, -eye[2]],
            [0, 0, 0, 1]
        ])

        lookat = np.matmul(R, E)
        far = GL.far
        near = GL.near
        
        top = near * np.tan(fieldOfView/2)
        aspect = GL.width / GL.height
        right = top * aspect
        bottom = -top
        left = -right
        perspective = np.array([
            [near / right, 0, 0, 0],
            [0, near / top, 0, 0],
            [0, 0, - (far + near) / (far - near), -2 * far * near / (far - near)],
            [0, 0, -1, 0]
        ])

        screen_transform = np.array([
            [GL.width / 2, 0, 0, GL.width / 2],
            [0, -GL.height / 2, 0, GL.height / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        GL.camera_transform = lookat
        GL.camera_perspective_transform = np.matmul(perspective, lookat)
        GL.perspective_transform = perspective
        GL.screen_transform = screen_transform
        m = np.matmul(screen_transform, GL.camera_perspective_transform)
        GL.viewpoint_transform = m


    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # Build local matrices
        t_mat = np.identity(4)
        s_mat = np.identity(4)
        r_mat = np.identity(4)
        t_mat[:3, 3] = translation
        s_mat[:3, :3] = np.diag(scale)
        axis = rotation[:3]
        angle = rotation[3]
        r_mat = GL.rotation_mat(axis, angle)
        m_local = np.matmul(t_mat, np.matmul(r_mat, s_mat))
        m_parent = GL.transform_stack[-1] if len(GL.transform_stack) > 0 else np.identity(4)
        m = np.matmul(m_parent, m_local)
        
        GL.transform_stack.append(m)

    @staticmethod
    def rotation_mat(axis, angle):
  
        axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.identity(4)
        ux, uy, uz = axis / norm
        half = angle / 2.0
        sin_half = np.sin(half)
        qx = ux * sin_half
        qy = uy * sin_half
        qz = uz * sin_half
        qr = np.cos(half)
        transform = np.array([[1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qr), 2*(qx*qz + qy*qr), 0],
                                  [2*(qx*qy + qz*qr), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qr), 0],
                                  [2*(qx*qz - qy*qr), 2*(qy*qz + qx*qr), 1 - 2*(qx**2 + qy**2), 0],
                                  [0, 0, 0, 1]])
                              
        return transform

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        print("TriangleStripSet : ")

        strip_start = 0
        transform = GL.transform_stack[-1]
        viewpoint_transform = GL.viewpoint_transform
        transform = viewpoint_transform @ transform
        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]
        for strip_size in stripCount:
            for triangle_i in range(strip_start, strip_start + (strip_size-2)*3,3):
                starting_coord = triangle_i 
                a = [point[starting_coord], point[starting_coord + 1], point[starting_coord + 2]]
                b = [point[starting_coord + 3], point[starting_coord + 4], point[starting_coord + 5]]
                c = [point[starting_coord + 6], point[starting_coord + 7], point[starting_coord + 8]]
                area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if area2 < 0:
                    b, c = c, b  # enforce CCW winding
                GL.draw_triangle(a, b, c, transform, color)
            strip_start += (strip_size*3)


    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        print("IndexedTriangleStripSet : ")
        # print(point)
        # print(index)
        # Pre-calculate the composed transform matrix
        composed_transform = GL.viewpoint_transform @ GL.transform_stack[-1]
        color = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])]
        diffuseColor = colors.get("diffuseColor", [1, 1, 1])
        
        # split index in i -1

        indexes = [[]]
        for i in index:
            if i == -1:
                indexes.append([])
            else:
                indexes[-1].append(i)
        indexes = indexes[:-1]
        for index_l in indexes:
            odd = False

            for vertex_i in range(0, len(index_l)-3):
                a_i = index_l[vertex_i] * 3
                b_i = index_l[vertex_i + 1] * 3
                c_i = index_l[vertex_i + 2] * 3
                if odd:
                    b_i, c_i = c_i, b_i
                odd = not odd
                
                # Extract triangle vertices as homogeneous coordinates
                a = np.array([point[a_i], point[a_i + 1], point[a_i + 2], 1])
                b = np.array([point[b_i], point[b_i + 1], point[b_i + 2], 1])
                c = np.array([point[c_i], point[c_i + 1], point[c_i + 2], 1])

                normal = None
                camera_space_coords = None
                
                if GL.headlight or GL.directionalLightEnabled:
                    # Calculate world space coordinates
                    a_world = GL.transform_stack[-1] @ a
                    b_world = GL.transform_stack[-1] @ b
                    c_world = GL.transform_stack[-1] @ c

                    # Calculate camera space coordinates for lighting
                    camera_space_a = GL.camera_transform @ a_world
                    camera_space_b = GL.camera_transform @ b_world
                    camera_space_c = GL.camera_transform @ c_world
                    camera_space_a = camera_space_a[:3] / camera_space_a[3]
                    camera_space_b = camera_space_b[:3] / camera_space_b[3]
                    camera_space_c = camera_space_c[:3] / camera_space_c[3]
                    
                    # Calculate normal in camera space
                    normal = -np.cross(camera_space_b[:3] - camera_space_a[:3], camera_space_c[:3] - camera_space_a[:3])
                    normal = normal / np.linalg.norm(normal)
                    
                    camera_space_coords = [camera_space_a, camera_space_b, camera_space_c]
                    color = diffuseColor
            
                # Apply transform to get screen coordinates
                a = composed_transform @ a
                b = composed_transform @ b
                c = composed_transform @ c
                a = a[:3] / a[3]
                b = b[:3] / b[3]
                c = c[:3] / c[3]
                if GL.headlight or GL.directionalLightEnabled:
                    GL.draw_triangle_2d(a, b, c, normal=normal, material=colors, camera_space_coords=camera_space_coords)
                else:
                    GL.draw_triangle_2d(a, b, c, color=color, transparency=colors.get("transparency", None))


    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        # print("IndexedFaceSet : ")
        # if coord:
        #     print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        # print("colorPerVertex = {0}".format(colorPerVertex))
        # if colorPerVertex and color and colorIndex:
        #     print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        # if texCoord and texCoordIndex:
        #     print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        print("IndexedFaceSet : ")
        image = None
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            # print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores
        coords_lists = [[]]
        transform = GL.viewpoint_transform @ GL.transform_stack[-1]

        coords_np = np.array(coord, dtype=float).reshape(-1, 3)         # [N,3]
        ones = np.ones((coords_np.shape[0], 1), dtype=float)            # [N,1]
        coords4 = np.hstack([coords_np, ones])                          # [N,4]

        # Screen-space transform for XY (includes perspective and screen mapping)
        clip = coords4 @ transform.T                                    # [N,4]
        clip /= clip[:, [3]]                                            # normalize
        coords_xy = clip[:, :3]

        # Camera-space Zs (linear in eye space)
        # I do this again even tough viewpoint has the same transform 
        # I could brake down the previous viewpoint transform but the number of computations is almost the same
        # this Way I  dont mess with parts of the code that are already working aand avoid potential references issues
        model_transform = GL.transform_stack[-1]
        cam_transform = GL.camera_transform @ model_transform
        cam_coords = coords4 @ cam_transform.T                           # [N,4]
        cam_z = cam_coords[:, 2]
        # Build per-vertex color list grouped every 3 values (r, g, b)
        vertexColors = None
        if colorPerVertex and color and colorIndex:
            try:
                vertexColors = np.array(color, dtype=float).reshape(-1, 3)
                print("vertexColors")
                print(vertexColors)
            except ValueError:
                vertexColors = None
        singleColor = [int(v * 255) for v in colors.get("emissiveColor", [1, 1, 1])] if vertexColors is None else None
        for coord_i in coordIndex:
            if coord_i == -1:
                coords_lists.append([])
            else:
                coords_lists[-1].append(coord_i)
        # Build per-vertex UVs if provided
        uv_points = None
        if texCoord:
            try:
                uv_points = np.array(texCoord, dtype=float).reshape(-1, 2)
            except ValueError:
                uv_points = None

        # Camera space coordinates for lighting calculations
        cam_coords_3d = cam_coords[:, :3] / cam_coords[:, [3]]  # [N,3] normalized camera space coords
        
        for face in coords_lists:
            if len(face) < 3:
                continue
            a = coords_xy[face[0]]
            color_a = vertexColors[face[0]] if vertexColors is not None else None
            uv_a = uv_points[face[0]] if uv_points is not None else None
            for i in range(1, len(face)-1):
                b = coords_xy[face[i]]
                color_b = vertexColors[face[i]] if vertexColors is not None else None
                c = coords_xy[face[i+1]]
                color_c = vertexColors[face[i+1]] if vertexColors is not None else None
                uv_b = uv_points[face[i]] if uv_points is not None else None
                uv_c = uv_points[face[i+1]] if uv_points is not None else None
                za = cam_z[face[0]]
                zb = cam_z[face[i]]
                zc = cam_z[face[i+1]]
                area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if area2 > 0: #Has to be greater instead of less because of the axis flip in transform
                    b, c = c, b  # enforce CCW winding
                    color_b, color_c = color_c, color_b
                    zb, zc = zc, zb
                    uv_b, uv_c = uv_c, uv_b
                
                # Calculate lighting parameters if needed
                normal = None
                camera_space_coords = None
                if GL.headlight or GL.directionalLightEnabled:
                    # Get camera space coordinates for this triangle
                    camera_space_a = cam_coords_3d[face[0]]
                    camera_space_b = cam_coords_3d[face[i]]
                    camera_space_c = cam_coords_3d[face[i+1]]
                    
                    # Swap if winding was corrected
                    if area2 > 0:
                        camera_space_b, camera_space_c = camera_space_c, camera_space_b
                    
                    # Calculate normal in camera space
                    edge1 = camera_space_b - camera_space_a
                    edge2 = camera_space_c - camera_space_a
                    normal = np.cross(edge1, edge2)
                    normal_length = np.linalg.norm(normal)
                    if normal_length > 0:
                        normal = normal / normal_length
                    else:
                        normal = np.array([0, 0, 1])  # Default normal
                    
                    camera_space_coords = [camera_space_a, camera_space_b, camera_space_c]
                
                triVertexColors = [color_a, color_b, color_c]
                triZs = [za, zb, zc]
                triUVs = None
                if uv_a is not None and uv_b is not None and uv_c is not None and current_texture:
                    triUVs = [uv_a, uv_b, uv_c]
                
                # Call draw_triangle_2d with lighting support
                if GL.headlight or GL.directionalLightEnabled:
                    GL.draw_triangle_2d(a, b, c, singleColor, triVertexColors, zs=triZs, uvs=triUVs, 
                                      texture=image if current_texture else None, normal=normal, 
                                      material=colors, camera_space_coords=camera_space_coords)
                else:
                    GL.draw_triangle_2d(a, b, c, singleColor, triVertexColors, zs=triZs, uvs=triUVs, 
                                      texture=image if current_texture else None, transparency=colors.get("transparency", None))


    

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        GL.headlight = headlight
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.
        GL.directionalLightEnabled = True
        GL.directionalLight_ambientIntensity = ambientIntensity
        GL.directionalLight_color = color
        GL.directionalLight_intensity = intensity
        GL.directionalLight_direction = [v for v in direction]
        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        #hermite interpolation
        previous_key = 0
        next_key = 0
        for k in range(len(key)):
            if set_fraction >= key[k]:
                previous_key = k
            if set_fraction < key[k]:
                next_key = k
                break
        #reshape keyVlaue to split in 3d vectors
        key_value = np.array(keyValue).reshape(-1, 3)
        print(f"set_fraction: {set_fraction}")
        print(f"previous_key: {previous_key}")
        print(f"next_key: {next_key}")
        # if previous_key == next_key:
        #     return keyValue[previous_key]
        
        s = (set_fraction - key[previous_key]) / (key[next_key] - key[previous_key])
        S = np.array([s**3, s**2, s, 1])
        H = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
        v1 = np.array(key_value[previous_key])
        v2 = np.array(key_value[next_key])
        t1 = (key_value[previous_key + 1] - key_value[previous_key-1 ])/2
        if next_key == len(key_value)-1:
            t2 = (key_value[1] - key_value[next_key-1])/2
        else: 
            t2 = (key_value[next_key + 1] - key_value[next_key-1 ])/2
        c = np.array([v1,v2,t1,t2])

        print(f"t1: {t1}")
        print(f"t2: {t2}")

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        # value_changed =    S @ H @ c
        value_changed = np.matmul(S, np.matmul(H, c))


        print(f"value_changed: {value_changed}")
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.
        # print(f"rot set_fraction: {set_fraction}")
        # print(f"rot key: {key}")
        # print(f"rot keyValue: {keyValue}")
        previous_key = 0
        next_key = 0

        values = np.array(keyValue).reshape(-1, 4)
     
        for k in range(len(key)):
            if set_fraction >= key[k]:
                previous_key = k
            if set_fraction < key[k]:
                next_key = k
                break
        prev = values[previous_key]
        nextv = values[next_key]
        s = (set_fraction - key[previous_key]) / (key[next_key] - key[previous_key])
        print(f"rot prev: {prev}")
        print(f"rot nextv: {nextv}")
        print(f"rot s: {s}")
        value =  [prev[0], prev[1], prev[2],GL.lerp(prev[3], nextv[3], s)]
        print(f"rot value: {value}")
        return value
        
    @staticmethod
    def lerp(v1, v2, s):
        return v1 + s * (v2 - v1)
    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
