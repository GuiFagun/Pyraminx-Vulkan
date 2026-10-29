#version 450

// primeiramente push constant é um espaco de memoria logico que define no layout do pipeline, ele determina quais bytes envia pros shaders e quais estagios podem acessar esses bytes. dai eu defino uma regiao de push constants pra receber esses dados que podem ser atualizados por comandos na hora de desenhar o frame
layout(push_constant) uniform PushConstants {
    mat4 mvp; // basicamente nesse caso aqui eu vou disponibilizar uma matriz mvp (modelo visao projecao) pra transformar os verticez durante a renderizaçao
} pc;

// vertices vindo de um vertex buffer
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    fragColor = inColor;
}