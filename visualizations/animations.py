from manim import *
import numpy as np

class PistaCircular(Scene):
    def construct(self):
        # Adiciona um texto à cena
        # text = Text("Olá, Manim!", font_size=24)
        # self.play(Write(text))

        # Adiciona um círculo à cena e o anima
        outer_circle = Circle(radius = 2.5)
        medium_circle = Circle(radius = 2.25).set_stroke(color=WHITE, width=2)
        medium_circle = DashedVMobject(medium_circle).set_dash_pattern([0.1, 0.1])
        inner_circle = Circle(radius = 2)

        # Primeira parte da cena, cria a pista 
        self.play(Create(outer_circle), Create(inner_circle), Create(medium_circle))
        self.wait(1)
    
        # Segunda parte da cena, hilight no circulo maior e depois no menos
        highlight_circle = outer_circle.copy().set_stroke(color=YELLOW, width=4)
        self.play(Create(highlight_circle))
        self.wait(1)
        self.play(Uncreate(highlight_circle))
        self.wait(1)

        highlight_circle = inner_circle.copy().set_stroke(color=YELLOW, width=4)
        self.play(Create(highlight_circle))
        self.wait(1)
        self.play(Uncreate(highlight_circle))
        self.wait(1)

        # Terceira parte da cena, cria um 'Carro' em cada uma das partes da pista e faz eles andarem com a mesma velocidade

        inner_car = Dot()
        outer_car = Dot()

        tracker = ValueTracker(0)

        def car_position(r, v, t):
            x = r * np.sin((v / ( 2 * PI * r)) * t)
            y = r * np.cos((v / ( 2 * PI * r)) * t)
            return [x, y, 0]

        # Associe a posição do ponto ao ValueTracker usando um updater
        inner_car.add_updater(lambda m: m.move_to(car_position(2 , 5, tracker.get_value())))
        outer_car.add_updater(lambda m: m.move_to(car_position(2.5 , 5, tracker.get_value())))
        self.add(inner_car)
        self.add(outer_car)

        anim_time = 4 * PI

        self.play(Create(inner_car), Create(outer_car))
        self.wait(1)

        self.play(tracker.animate.set_value(2 * PI * anim_time), run_time=10, rate_func=linear)

        self.wait(2)


        self.play(*[FadeOut(mob)for mob in self.mobjects])
        self.wait(1)


class PistaMenorCaminho(Scene):
    def construct(self):
        
        scale = 1/30
        deslocamento = [-1.5, 3, 0]
        base = np.loadtxt('resources/Baseline.txt').T

        centerline = np.column_stack([base[1], base[0], np.zeros(np.size(base[0]))]) * scale + deslocamento
        left_border = np.column_stack([base[3], base[2], np.zeros(np.size(base[2]))]) * scale + deslocamento
        right_border = np.column_stack([base[5], base[4], np.zeros(np.size(base[4]))]) * scale + deslocamento
        shortest_path = np.column_stack([base[7], base[6], np.zeros(np.size(base[4]))]) * scale + deslocamento

        # Crie uma curva de Bezier suavizando entre os pontos
        curve_center = VMobject()
        curve_center.set_points_as_corners(centerline)
        curve_center.set_stroke(width = 1)
        curve_center = DashedVMobject(curve_center, num_dashes=100)

        curve_left = VMobject()
        curve_left.set_stroke(width=2)
        curve_left.set_color(YELLOW)
        curve_left.set_points_as_corners(left_border)

        curve_right = VMobject()
        curve_right.set_stroke(width=2)
        curve_right.set_color(YELLOW)
        curve_right.set_points_as_corners(right_border)

        curve_sp = VMobject()
        curve_sp.set_points_as_corners(shortest_path)
        curve_sp.set_color(RED)
        curve_sp.set_stroke(width=5)
        curve_sp = DashedVMobject(curve_sp, num_dashes=200)

        self.add(curve_center)
        self.add(curve_left)
        self.add(curve_right)

        self.play(Create(curve_center), Create(curve_right), Create(curve_left), run_time=2)
        self.wait(2)

        self.add(curve_sp)
        self.play(Create(curve_sp), run_time = 2)
        self.wait(2)

        car = Dot(color=WHITE)
        car.stroke_width = 10
        car.move_to(shortest_path[0])

        self.add(car)
        self.play(Create(car))

        tracker = ValueTracker(0)
        car.add_updater(lambda m: m.move_to(shortest_path[np.clip(int(tracker.get_value() * np.size(shortest_path)), 0, 499)]))

        self.play(tracker.animate.set_value(0.4), run_time=5, rate_func = linear)

        circles = \
        [
            Circle(radius=1.5, color=GREEN).move_to([5, 2, 0]),
            Circle(radius=1.4, color=GREEN).move_to([5.6, -2, 0]),
            Circle(radius=1.7, color=GREEN).move_to([-2.1, -2.3, 0]),
        ]

        self.play(*[Create(i) for i in circles])
        self.wait(2)

        self.play(*[FadeOut(mob)for mob in self.mobjects])
        self.wait(1)


from manim import *

class SemicircleScene(Scene):
    def construct(self):
        # Raio do semicírculo
        radius = 3
        center = [-3, -1, 0]

        # Crie um semicírculo
        semicircle = Arc(start_angle= (2/3) * PI, angle = PI/2, radius=radius, arc_center=center, color=BLUE)

        # Adicione o semicírculo à cena
        self.add(semicircle)

        # Adicione o centro do semicírculo à cena
        center_dot = Dot(center, color=WHITE)
        self.add(center_dot)

        # Adicione um ponto no semicírculo
        point_on_circle = Dot(semicircle.point_from_proportion(0.5), color=RED)
        point_on_circle.set_stroke(width=6)
        radius_line = DashedLine(semicircle.point_from_proportion(1), center)
        self.add(point_on_circle)

        # Crie um vetor apontando para o centro do semicírculo a partir do ponto no círculo
        vector_to_center = Arrow(point_on_circle.get_center(), (center + point_on_circle.get_center())/2, buff=0.1, color=GREEN)
        # Calcule o vetor raio que passa pelo centro do círculo até o ponto escolhido
        radius_vector = Arrow(point_on_circle.get_center(), (center + point_on_circle.get_center())/2, buff=0.1, color=YELLOW)

        # Calcule o vetor tangente como o vetor perpendicular ao vetor raio
        tangent_vector = radius_vector.copy().rotate(PI / 2, about_point=point_on_circle.get_center())

        # Adicione o vetor à cena
        self.add(vector_to_center)

        radius_label = MathTex(r"R", color=WHITE).next_to(radius_line, direction=DOWN, buff=-0.3)
        tangent_label = MathTex(r"v", color=YELLOW).next_to(tangent_vector, direction=UP, buff=0.2)
        center_label = MathTex(r"F_{cp}", color=GREEN).next_to(radius_vector, direction=DOWN, buff=0)
        mass_label = MathTex(r"m", color=RED).next_to(point_on_circle, direction=DOWN *0.9 + LEFT, buff=0.1)

        fcp_formula = MathTex(r"F_{cp} = \frac{mv}{R}", color=WHITE).scale(1.7).move_to([0.5, -0.5, 0])

        

        # Animação da cena

        self.camera.frame_width = 16/1.5
        self.camera.frame_height = 9/1.5
        self.camera.frame_center = [-1.7, -0.6, 0]


        self.play(Create(semicircle),Create(tangent_vector), Create(center_dot), Create(point_on_circle), Create(vector_to_center), Create(radius_line),  run_time=2)
        self.play(Create(radius_label), Create(tangent_label), Create(mass_label), Create(center_label))

        self.wait(2)

        self.play(Create(fcp_formula))
        
        self.wait(2)


        self.play(*[FadeOut(mob)for mob in self.mobjects])
        self.wait(1)


class GeneScene(Scene):
    def construct(self):
        scale = 1/35
        deslocamento = [0, 2.5, 0]
        base = np.loadtxt('resources/Baseline.txt').T

        centerline = np.column_stack([base[1], base[0], np.zeros(np.size(base[0]))]) * scale + deslocamento
        left_border = np.column_stack([base[3], base[2], np.zeros(np.size(base[2]))]) * scale + deslocamento
        right_border = np.column_stack([base[5], base[4], np.zeros(np.size(base[4]))]) * scale + deslocamento
        shortest_path = np.column_stack([base[7], base[6], np.zeros(np.size(base[6]))]) * scale + deslocamento
        minimum_curvature = np.column_stack([base[9], base[8], np.zeros(np.size(base[8]))]) * scale + deslocamento
        

        # Crie uma curva de Bezier suavizando entre os pontos
        curve_center = VMobject()
        curve_center.set_points_as_corners(centerline)
        curve_center.set_stroke(width = 1)
        curve_center = DashedVMobject(curve_center, num_dashes=100)

        curve_left = VMobject()
        curve_left.set_stroke(width=2)
        curve_left.set_color(YELLOW)
        curve_left.set_points_as_corners(left_border)

        curve_right = VMobject()
        curve_right.set_stroke(width=2)
        curve_right.set_color(YELLOW)
        curve_right.set_points_as_corners(right_border)

        curve_sp = VMobject()
        curve_sp.set_points_as_corners(shortest_path)
        curve_sp.set_color(RED)
        curve_sp.set_stroke(width=5)
        curve_sp = DashedVMobject(curve_sp, num_dashes=200)

        curve_mc = VMobject()
        curve_mc.set_points_as_corners(minimum_curvature)
        curve_mc.set_color(GREEN)
        curve_mc.set_stroke(width=5)
        curve_mc = DashedVMobject(curve_mc, num_dashes=200)

        self.add(curve_center)
        self.add(curve_left)
        self.add(curve_right)
        
        mask = np.array(base[10], dtype=int)
        np.random.seed(0)
        alp = (np.random.random(size=mask[-1]) * 0.1 - 0.05) + 0.5

        def new_curve(alpha):
            new_alpha = alpha[mask - 1]
            new_alpha = np.column_stack((new_alpha, new_alpha, new_alpha))
            return shortest_path * new_alpha + minimum_curvature *(1 - new_alpha)

        animated_curve = VMobject()
        animated_curve.set_points_as_corners(new_curve(alp))
        animated_curve.set_color(BLUE)
        animated_curve.set_stroke(width=8)

        self.play(Create(curve_center), Create(curve_right), Create(curve_left), run_time=2)

        self.wait(2)

        self.play(Create(curve_sp), Create(curve_mc), run_time = 2)

        curves = [ curve_right, curve_left]
        
        self.wait(2)

        matriz = Matrix([[round(i , 2)] for i in alp])
        matriz.scale(0.6)
        matriz.set_stroke(width=2)
        matriz.set_color(BLUE)
        matriz.move_to([-5.5, -0, 0])

        self.play(Create(animated_curve), Create(matriz), run_time = 2)

        self.wait(2)

        tracker = ValueTracker(0)
        animated_curve.add_updater(lambda m : m.set_points_as_corners(new_curve(alp + np.sin(tracker.get_value())/2)))

        self.play(tracker.animate.set_value(2 * PI), run_time = 10)

        self.wait(2)   
        
        self.play(*[FadeOut(mob)for mob in self.mobjects])  

        self.wait(2)
        
        qtd_matricez = 6

        matrices = [np.random.random(size=mask[-1]) for i in range(qtd_matricez)]
        matrices_render = [Matrix([[j.round(2)] for j in i]) for i in matrices]

        initial_position = matriz.get_center() + RIGHT * 0.6 + UP * 0.1
        separation = 2

        matrices_render = [matrices_render[i].move_to(initial_position + RIGHT * separation * i).scale(0.6) for i in range(len(matrices_render))]

        results = [np.random.random() + 45.0 for i in range(qtd_matricez)]
        results_render = [Text(str(round(i, 2)) + 's').scale(0.4) for i in results]

        initial_position = initial_position + DOWN * 3.6
        results_render = [results_render[i].move_to(initial_position + RIGHT * separation * i) for i in range(qtd_matricez)]

        self.play(*[Create(i) for i in matrices_render])
        self.wait(1)
        self.play(*[Create(i) for i in results_render])
        self.wait(1) 

        for k in range(4):
            ind_menor =  results.index(min(results))
            mtr_cp = matrices_render[ind_menor].copy().set_color(YELLOW).set_stroke(width = 1)
            tex_cp = results_render[ind_menor].copy().set_color(YELLOW).set_stroke(width = 1)

            self.play(Create(mtr_cp), Write(tex_cp))

            mob_state = self.mobjects

            self.wait(1)

            matrices = [np.random.random(size=mask[-1]) for i in range(qtd_matricez)]
            matrices_render = [Matrix([[j.round(2)] for j in i]) for i in matrices]

            initial_position = matriz.get_center() + RIGHT * 0.6 + UP * 0.1
            separation = 2

            matrices_render = [matrices_render[i].move_to(initial_position + RIGHT * separation * i).scale(0.6) for i in range(len(matrices_render))]

            results = [np.random.random() + 45.0 for i in range(qtd_matricez)]
            results_render = [Text(str(round(i, 2)) + 's').scale(0.4) for i in results]

            initial_position = initial_position + DOWN * 3.6
            results_render = [results_render[i].move_to(initial_position + RIGHT * separation * i) for i in range(qtd_matricez)]

            self.play(*[ Transform(mtr_cp.copy(), i) for i in matrices_render], 
            *[FadeOut(mob)for mob in mob_state])

            self.play(*[Write(i) for i in results_render])

            self.wait(1)      
        
        ind_menor =  results.index(min(results))
        mtr_cp = matrices_render[ind_menor].copy().set_color(YELLOW).set_stroke(width = 1)
        tex_cp = results_render[ind_menor].copy().set_color(YELLOW).set_stroke(width = 1)

        self.play(Create(mtr_cp), Write(tex_cp))

        self.wait(1)
        
        self.play(*[FadeOut(mob)for mob in self.mobjects])  
        self.wait(2)


# Executa a animação
if __name__ == "__main__":
    scene = PistaCircular()
    scene.render()
