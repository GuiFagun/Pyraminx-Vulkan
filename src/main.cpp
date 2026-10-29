#include "glm/fwd.hpp"
#include <string>
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

/*
Projeto para o 7 semestre da universidade para a matéria de
computação gráfica. O projeto, em teoria, foi pensado para ser feito no Unity,
porém como eu me interesso bastante pela área e queria aprender mais, decidi
fazer "no bruto" e tentar aprender num tipo de desafio "teste de choque" e
entender ao máximo fundamentalmente como funcionar um programa de computacao
grafica usando a API Vulkan. Pra ser uma boa referencia futura, talvez, EU
escrevi bastante comentario durante o código, como se fosse anotaço de caderno
mesmo, isso porque cada parte dele demorou bastante pesquisando, consultando
documentação, assistindo tutorial e corrigindo problemas/tirando dúvidas com IA,
e como esse projeto é primariamente um objeto de ESTUDO, eu quis deixar nele
comentários (muitos de forma coloquial) pra que seja possivel voltar nele e
relembrar estruturas comuns da linguagem e da API gráfica, além de que impede
que você saia replicando o que você viu em algum tutorial, vídeo, fórum, ou IA
sem entender o que ta fazendo.
*/

// basicamente isso aqui evita problemas entre a comunicqção do driver da gpu
// com o programa. Porque garante que o compilador interprete um unsigned int
// sempre como 4 bytes. Se não fosse fixo, o número de bytes que o driver espera
// poderia divergit do que o compilador gerou.

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// Essas layers de validaçao do vulkan evita que caso eu cometa algum erro ou
// algum problema aconteça, não trave ou apresente comportamento errado sem
// avisar. O vulkan vai olhar cada chamada e imprimir mensagem de erro.

const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// SWAPCHAIN é a fila de imagens que revezam entre ser desenhada pela gpu e
// sendo mostrada na tela. É exatamente a famosa questão de como a gpu evita o
// tearing, porque enquanto uma ta sendo mostrada, ele ja ta desenhando a outra.
// Precisa ser criada diretamente pelo programa.
const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

// a placa de video pode ter diferentes filas de trabalho, o programa precisa
// perguntar pra gpu quais dessas filas ela tem e escolher qual vai usar.
// uma FILA é basicamente um ponto de entrada singular pra mandr comandos pra
// gpu. Voce grava os comandos no buffer de comando (CPU escrevendo uma lista de
// comando), entrega pra fila e aí a gpu executa o que foi entregue pra fila (e
// nisso a cpu segue executando o resto do programa) ja a FAMILIA sao varias
// filas que tem as mesmas habilidades, essas familias definem as FLAGS (o que
// as filas sabem fazer), a QUANTIDADE de filas e o INDICE da familia (posicao
// dela na lista de familias)
struct QueueFamilyIndices {

  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;

  bool isComplete() {
    return graphicsFamily.has_value() && presentFamily.has_value();
  }
};

// so pra agrupar tudo que a gpu e a janela suportam
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

// VERTICE, essenciamente um conjunto de atriibutos que descreve um ponto na
// construçao de uma geometria.
struct Vertex {
  glm::vec3 pos;   // posicao do vertide, composta de xyz
  glm::vec3 color; // cor do vertice, composta de rgb

  // quando ta mexendo com um buffer de vertices voce precisa explicar COMO esse
  // buffer vai ser dividido em vertices
  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride =
        sizeof(Vertex); // numero de bytes entre um vértice e o próximo
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return binding;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attrs{};
    attrs[0].binding = 0;
    attrs[0].location = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = offsetof(Vertex, pos);

    attrs[1].binding = 0;
    attrs[1].location = 1;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = offsetof(Vertex, color);
    return attrs;
  }
};

constexpr float EDGE = 1.2f; // tamanho da aresta

// vertices (x,0,0), (-d,+-a/2,0), (0,0,h)
// o a é a aresta
// x e y sendo 0, a origem fica na base da piramide, mas z=0 deixa na linha da
// base no plano z, entao tem que subtrair pelo r (distancia do centro pra face)
// pra que a origem fique bem no meio da piramide
std::array<glm::vec3, 4> tetraCorners() {
  float a = EDGE;
  float x = std::sqrt(3.0f) / 3.0f * a;
  float r = std::sqrt(6.0f) / 12.0f * a;
  float d = std::sqrt(3.0f) / 6.0f * a;
  float h = std::sqrt(2.0f / 3.0f) * a;
  return {glm::vec3(x, 0.0f, -r), glm::vec3(-d, a / 2, -r),
          glm::vec3(-d, -a / 2, -r), glm::vec3(0.0f, 0.0f, h - r)};
}

// a cor fica no vertice, so que um vertice so pode ter uma cor, entao como cada
// vertice compartilha 3 lados e nao pode ter 3 cores ao mesmo tempo, eu coloco
// 3 vertices pra cada vertice, cada um tendo a cor de uma das faces que ele
// compartilha, quando juntar 3 vertices de pontos separados que compartilham
// uma mesma face e portanto tem a mesma cor, se interpolam e aquela face gera
// aquela cor.AUDIO NO GRUPO EXPLICANDO MELHOR, PQ É CONFUSO MESMO
std::vector<Vertex> makeTetrahedron() {
  auto p = tetraCorners();
  const glm::vec3 red(1, 0, 0), green(0, 0.8f, 0), blue(0, 0, 1),
      yellow(1, 0.9f, 0);
  return {
      {p[0], red},    {p[1], red},    {p[2], red}, // base
      {p[0], green},  {p[1], green},  {p[3], green},
      {p[1], blue},   {p[2], blue},   {p[3], blue},
      {p[2], yellow}, {p[0], yellow}, {p[3], yellow},
  };
}

const std::vector<Vertex> vertices = makeTetrahedron();

// a estrategia propyraminx que eu decidi foi dividir as faces do tetraedro
// principal nos 9 triangulos ao inves de renderizar vairos tetraedros em cada
// parte.
constexpr uint32_t PYRAMINX_TRIANGLES = 36;
constexpr uint32_t PYRAMINX_VERTEX_COUNT = PYRAMINX_TRIANGLES * 6;

struct Tri {
  glm::vec3 p[3];
  glm::vec3 color;
};

// buchosa demais pra explicar escrevendo, consultar audio
std::vector<Tri> makePyraminx() {
  auto c = tetraCorners();
  const glm::vec3 colors[4] = {
      {0, 0.8f, 0}, {1, 0, 0}, {0, 0.3f, 1}, {1, 0.9f, 0}};
  std::vector<Tri> out;

  for (int f = 0; f < 4; f++) {
    glm::vec3 V[3];
    int n = 0;
    for (int v = 0; v < 4; v++)
      if (v != f)
        V[n++] = c[v];

    auto P = [&](int i, int j) {
      return V[0] + (V[1] - V[0]) * (i / 3.0f) + (V[2] - V[0]) * (j / 3.0f);
    };
    auto add = [&](glm::vec3 a, glm::vec3 b, glm::vec3 d) {
      Tri t;
      t.p[0] = a;
      t.p[1] = b;
      t.p[2] = d;
      t.color = colors[f];
      out.push_back(t);
    };

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3 - i; j++) {
        add(P(i, j), P(i + 1, j), P(i, j + 1));
        if (i + j < 2)
          add(P(i + 1, j), P(i, j + 1), P(i + 1, j + 1));
      }
    }
  }
  return out;
}

// funçao pra ler arquivos .spv, que são binarios intermediarios de shaders. Os
// shaders .vert e .frag por exemplo vao ser convertidos pra SPIR-V pelo
// compilador
static std::vector<char> readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Erro na abertura do arquivo" + filename);
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

class PyraminxApp {

public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  VkInstance instance = nullptr;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE; // a gpu
  VkDevice device =
      VK_NULL_HANDLE; // a configuracao que o programa vai pedir da gpu
  VkQueue graphicsQueue =
      VK_NULL_HANDLE; // a fila grafica, pra onde manda os comandos de desenho
  VkQueue presentQueue =
      VK_NULL_HANDLE; // a fila de apresentacao, a que manda a imagem pra tela
  // um detalhe que vai aparecer quando depois, é que na maioria das gpus essas
  // duas filas sao a mesma, isso vai ser tratado quando formos apontar elas com
  // o vkDeviceQueue, mas vvai ser tratado como duas extamente pra funcionoar em
  // qualquer placa

  VkSurfaceKHR surface;

  VkSwapchainKHR swapChain;
  std::vector<VkImage> swapChainImages;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  std::vector<VkImageView> swapChainImageViews;

  // objetos que vao ligar as imagens da swapchain ao que vai ser desenhado
  VkRenderPass renderPass;         // o que o desenho usa e produz
  VkPipelineLayout pipelineLayout; // quais dados externos os shaders recebem
  VkPipeline graphicsPipeline;     // pipeline grafico inteiro já configurado

  std::vector<VkFramebuffer>
      swapChainFramebuffers; // liga as image views da swapchain ao render pass
  VkCommandPool commandPool; // gerenciador de memoria dos buffer de comando
  std::vector<VkCommandBuffer>
      commandBuffers; // lista de comandos antecipados, diferente do opengl que
                      // cada comando/chamada é executado na hora

  // na gpu as coisas quase tudo sempre é assíncrono, por isso é importante
  // trabalhar com tratamento de concorrência, nesse caso, semáforos
  VkSemaphore
      imageAvailableSemaphore; // so desenha quando a imagem tiver disponivel
  std::vector<VkSemaphore>
      renderFinishedSemaphores; // so mostra quando tiver desenhado (tive que
                                // mudar pra um array que vai ser um pra cada
                                // imagem, consultar audio que mandei no grupo
                                // solo explicando pq nao vou escrever nao)
  VkFence inFlightFence;        // cpu com gpu, a placa de vicdeo nao enxerga o
                                // semaforo, mas a fence sim

  // VERTEX BUFFER meus amigos, uma das partes fundamentais da renderizacao 3d,
  // basicamente um vkBuffer (handle que representa uma regiao linear na memoria
  // da placa) que armazena os dados dos vertices utilizados no pipeline
  // grafico.
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  glm::vec3 position{0.0f};
  glm::quat orientation{1.0f, 0.0f, 0.0f,
                        0.0f}; // quaternion identidade = sem rotação

  // o buffer de profundidade vai guardar pra cada pixel qual distancia da
  // camera ta o que foi desenhado nele. Se nao tiver esse buffer a placa de
  // video simplesmente vai desenhar na ordem que chegar (sempre ganhar o ultimo
  // que chega)
  VkImage depthImage;
  VkDeviceMemory depthImageMemory;
  VkImageView depthImageView;
  VkFormat depthFormat;

  void *mappedVertices = nullptr; // permanentemente mapeia o vertex buffer
  std::vector<Tri> pyraminx;      // estado do jogo
  std::vector<char> moving;       // que peças tao na camada atual
  bool animating = false;
  glm::vec3 moveAxis{0.0f}; // eeixo da jogada atual
  float moveAngle = 0.0f;
  float moveTarget = 0.0f;
  bool keyWasDown[8] = {}; // uma jogada so por teclada

  int sceneMode = 0; // tetraedro, tres tetraedos, pyraminx

  // pra entender isso aqui e as outras rotaçoes tambem tipo la encima na do
  // tetraedro solo, precisa entender oq é um quarternion. Basicamnete sao 4
  // numeros, xyzw, que representam uma rotacao em torno de um eixo por um
  // angulo especifico. w = cos(angulo/2) xyz = sen((angulo/2)*eixo de rotacao)

  std::vector<glm::mat4> buildLocalTransforms() {

    if (sceneMode != 1)
      return {glm::mat4(1.0f)};

    const float a = EDGE;
    const float d = std::sqrt(3.0f) / 6.0f * a;

    const glm::vec3 zAxis(0, 0, 1);
    const glm::quat flipY =
        glm::angleAxis(glm::radians(180.0f), glm::vec3(0, 1, 0));
    const glm::quat up = glm::angleAxis(glm::radians(-90.0f), zAxis) * flipY;
    const glm::quat down = glm::angleAxis(glm::radians(90.0f), zAxis) * flipY;

    const glm::vec3 vet(a, 0, 0);
    const glm::vec3 center(-a / 2, -d / 2, 0);
    const glm::vec3 p1 = center;
    const glm::vec3 p2 = center + glm::vec3(a / 2, d, 0);
    const glm::vec3 p3 = center + vet;

    auto place = [](glm::vec3 p, glm::quat q) {
      return glm::translate(glm::mat4(1.0f), p) * glm::mat4_cast(q);
    };

    return {place(p1, up), place(p2, down), place(p3, up)};
  }

  void initWindow() {

    glfwInit();

    // Ele vai tentar abrir um contexto de opengl, tem que deixar claro pra nao
    // fazer isso (o glwf foi feito, por padrao, pro opengl)

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // redimensionar a janela precisa recriar a swapchain toda. Na parte inicial
    // aqui vou manter travado, daí depois que eu tiver o swapchain zero bala
    // vou voltar aqui e indicar que mudei, mas fica o registro. [swapchain:
    // fila de imagens que a placa vai desenhar e mostrar na tela]
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window =
        glfwCreateWindow(WIDTH, HEIGHT, "Pyraminx Vulkan", nullptr, nullptr);
  }

  void initVulkan() {
    createInstance(); // conexao entre o programa e a biblioteca vulkan, dizendo
                      // o nome, versao, extensoes e camadas de validacao
    createSurface();  // ponte entre o vulkan e a janela do glwf
    pickPhysicalDevice();  // escolher a gpu que vai ser usada
    createLogicalDevice(); // cria o dispositivo logico, que é a conexão do
                           // programa com a gpu fisica
    createSwapchain();     // cria a ffila de imagens  vai mostrar na tela,
                           // decididnod o formato, tamamho, apresentaçao e
                           // quantidade
    createImageViews();    // o swapchain retorna VkImage, que é so um bloco de
                           // memoria inrenderizavel. A image view diz COMO
                           // interpretar esse bloco. Criada para cada imagem da
                           // swapchain.
    createRenderPass(); // define as regras das imagens que vao ser desenhadas,
                        // como por exemplo a cor da swapchain e o que fazer com
                        // as imagens no inicio/fim
    createGraphicPipeline(); // monta o pipeline grafico com suas devidas
                             // configurações
    createDepthResources();  // imagem de profundidade pro pipeline usar pra
                             // jogar fora picels que tao atras de outros
    createFramebuffers();    // cria um framebuffer pra cada image view
    createCommandPool();     // crial a pool que vai recegbervesses buffers de
                             // comando
    createVertexBuffer();    // cria o buffer dos vertices
    createCommandBuffers(); // aloca os buffers de comando nessa pool que acabou
                            // de criar (não grava os comandos ainda)
    createSyncObjects();    // cria os semáforos e a fence
  }

  void createInstance() {

    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error(
          "Camadas de validacao solicitadas mas indisponiveis. Verifique se o "
          "driver da GPU ta atualizado e se o Vulkan ta instalado "
          "corretamente.");
    }

    // Infos que o driver pode usar pra otmizar a execução do programa. Tipo
    // nome do app, versão, etc
    VkApplicationInfo appInfo{};
    appInfo.sType =
        VK_STRUCTURE_TYPE_APPLICATION_INFO; // esse campo aqui aparece direto
                                            // nas outras structs que sao
                                            // criadas e basicamente significa
                                            // indicar pro Vulkan QUAL struct
                                            // essa daqui é
    appInfo.pApplicationName = "Pyraminx Vulkan";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // O glwf sabe exatamente o que ele precisa pra desenhar a janela no OS,
    // aqui vou pedir uma lista pra ele em vez de ficar listando manualmente

    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;

    if (enableValidationLayers) {
      createInfo.enabledLayerCount =
          static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    // Toda funçao vulkan retorna esse vkresult, então por padrão tem que sempre
    // checar o vksuccess (se deu certo)

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("Erro ao tentar criar a instancia do Vulkan.");
    }

    std::cout << "Sucesso ao criar a instancia do Vulkan\n";
  }

  void createSurface() {
    // o proprio glfw sabe que funcao chamar pra casa os, entao nao precisa
    // implementar manualmente
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro ao criar a surface.");
    }
  }

  bool checkValidationLayerSupport() {

    // O vulkan nunca aloca memoria da saída que vai preencher, como nesse caso
    // nao da pra saber quantas layers vao ter, chama duas vezes
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(
        &layerCount,
        nullptr); // chama pra saber quantas layers existem usadno o nullptr

    std::vector<VkLayerProperties> availableLayers(
        layerCount); // aloca o array do tamanho certinho
    vkEnumerateInstanceLayerProperties(
        &layerCount, availableLayers.data()); // chama denovo pra preencher

    for (const char *layerName : validationLayers) {
      bool layerFound = false;

      for (const auto &layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }
    return true;
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {

    QueueFamilyIndices indices;

    // atençao denovo pro padrao de duas chamadas que aparece comummente
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());

    // cada tipo de fila (familia) tem flags indicando o que ela suporta, o
    // objetivo aqui é encotrar a VK_QUEUE_GRAPHICS_BIT
    int i = 0;
    for (const auto &queueFamily : queueFamilies) {
      if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphicsFamily = i;
      }

      // Verifica se essa familia de queue suporta apresentacao de imagens nessa
      // surface especifica
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

      if (presentSupport) {
        indices.presentFamily = i;
      }

      if (indices.isComplete()) {
        break;
      }
      i++;
    }
    return indices;
  }

  bool isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
      SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
      swapChainAdequate = !swapChainSupport.formats.empty() &&
                          !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
  }

  void pickPhysicalDevice() {

    // chamada dupla denovo
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
      throw std::runtime_error("Nenhuma GPU tem suporte ao Vulkan");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &dev : devices) {
      if (isDeviceSuitable(dev)) {
        physicalDevice = dev;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("Nenhuma GPU comptivel encontrada");
    }

    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    std::cout << "GPU: " << deviceProperties.deviceName << "\n";
  }

  // no Vulkan um dispositivo logico basicamente é a conexão da aplicação com a
  // GPU fisica, sendo uma instancia da implementação do dispositivo fisico com
  // seu proprio estado e recursos independentes de outros dispositivos lógicos.
  void createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    // remove duplicatas, porque pode ser que a mesma fila seja usada pra
    // graficos e pra apresentacao, dai nao precisa criar duas filas iguais
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};

    // a criação de um dispositivo logico ta diretamente associada com a criacao
    // das filas envolvendo este mesmo dispositivo, uma queue é basicamente o
    // canal pelo qual a aplicação envia buffers de comandos pra GPU pra
    // execuçao assincrona, ou seja, você escreve toda a ordem de comandos em um
    // buffer que será enviado até a GPU utilizando a queue, quando a GPU
    // recebe, a CPU volta a trabalhar enquanto a placa de vídeo trabalha no
    // fundo.
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // array das features da GPU que vao ser usadas
    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos =
        queueCreateInfos.data(); // array de structs de criação de fila
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(
        queueCreateInfos.size()); // numero de filas, utilizando o tamanho do
                                  // array que criou no loop antes
    createInfo.pEnabledFeatures =
        &deviceFeatures; // endereço doarray de features da GPU que vao ser
                         // usadas
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames =
        deviceExtensions.data(); // array de extensoes que vao ser usadas

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro criando o dispositivo logico");
    }

    // vai la pegar a fila que acabou de ser criada
    vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());

    // std::set com todos as extensoes que precisa, vai riscando cada uma que
    // achar daí se sobrar algum set no final é pq ta faltando
    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  // essa funcao aqui retorna uma struct com os detalhes do que a swapchain
  // suporta. É baseada no https://github.com/autun12/VulkanProgramming

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
    SwapChainSupportDetails details; // struct criada la em cima

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        device, surface,
        &details.capabilities); // limites basicaos como tamanho max e min,
                                // transformacoes suportadas, etc

    uint32_t formatCount; // quais formatos de cor sao aceitos
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         nullptr);
    if (formatCount != 0) {
      details.formats.resize(formatCount);
      vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                           details.formats.data());
    }

    uint32_t
        presentModeCount; // modos de apresentacao, ou seja como a swapchain vai
                          // controlar a exibicao e troca de imagens
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                                              &presentModeCount, nullptr);
    if (presentModeCount != 0) {
      details.presentModes.resize(presentModeCount);
      vkGetPhysicalDeviceSurfacePresentModesKHR(
          device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
  }

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // escolhe o formato de cor da swapchain baseado no que a placa de video
    // suporta. nesse caso o srgb que é o utilizado na maioria dos monitores
    for (const auto &availableFormat : availableFormats) {
      if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
          availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
        return availableFormat;
      }
    }

    // pega o primeiro que tiver se não encontrar o ideal
    return availableFormats[0];
  }

  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes) {
    // vai procurar se tem suporte pra MAILBOX, que é basicamente um triple
    // buffer, ou seja, tem o quadro mostrado, o que ta aguardando e um sendo
    // feito. Se esse novo for feito antes do que ta aguardando ser mostrado,
    // ele mostra o que acabou de ser feito e pula o que tava aguardando. Isso
    // evita o tearing e tem baixa latencia.
    for (const auto &availablePresentMode : availablePresentModes) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
        return availablePresentMode;
      }
    }

    // se nao tiver o MAILBOX usa o FIFO, first in first out, que é o basicao,
    // um faz o outro mostra, terminou de fazer mostra e comeca a fazer o outro
    // e é o unico com suporte garantido pelo vulkan
    return VK_PRESENT_MODE_FIFO_KHR;
  }

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {

    // decide o tamanho em pixels da imagem do swapchain, primeiramente vemos se
    // o sistema ja decidiu qual o tamanho e se sim vai com esse
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }

    // se o sistema deixar o programa escolher, dai cria o proprio tamanho aqui
    VkExtent2D actualExtent = {WIDTH, HEIGHT};
    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    return actualExtent;
  }

  void createSwapchain() {
    SwapChainSupportDetails swapChainSupport =
        querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat =
        chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode =
        chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    // pra evitar que a gpu fique esperando a aplicacao liberar uma imagem antes
    // de desenhar a proxima, coloca +1 no minimo
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 &&
        imageCount > swapChainSupport.capabilities.maxImageCount) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface; // qual janela que a swapchain vai apresentar
    createInfo.minImageCount =
        imageCount; // quantas imagens no minimo ela vai ter
    createInfo.imageFormat =
        surfaceFormat.format; // como cada pixel vai ser armazenado
    createInfo.imageColorSpace =
        surfaceFormat.colorSpace;    // como interpretar cor
    createInfo.imageExtent = extent; // largura e altura das imagens
    createInfo.imageArrayLayers =
        1; // numero de camadas da imagem, so passa de 1 em VR, que nao é o caso
    createInfo.imageUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // pra que vai usar as imagens,
                                             // nesse caso é pra renderizar
                                             // direto nelas

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
      // se tiver duas filas diferentes pra grafico e apresentaçao, a imagem
      // pode ser usada por mais de uma fila sem ficas transferindo quem ta com
      // a imagem
      createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      createInfo.queueFamilyIndexCount = 2;
      createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
      // so tem uma fila e so uma usa a imagem por vez
      createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform =
        swapChainSupport.capabilities
            .currentTransform; // campo pra dizer pro vulkan se ele deve aplicar
                               // algumaa transofrmacao na imagem tipo um
                               // celular que ta virado ou espelhado, como nesse
                               // caso nao ta, currentTransform, ou seja, deixa
                               // como ta
    createInfo.compositeAlpha =
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; // janela opaca e não transparente
    createInfo.presentMode = presentMode;  // modo de apresentaçao
    createInfo.clipped =
        VK_TRUE; // nao importa pixels escondidos, por exemplo se tiver outra
                 // janela na frente ele pode pular o desenho e trabalho deles
    createInfo.oldSwapchain =
        VK_NULL_HANDLE; // so vai ser importante se liberar alterar o tamanho da
                        // janela depois

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) !=
        VK_SUCCESS) {
      throw std::runtime_error("Nao conseguiu criar a swapchain");
    }

    // padrao duas chamadas denovo, dessa vez no final, porque se tiver criado
    // mais imagens do que o minimo estipulado, aqui a chamada dupla que já
    // define antes o tamanhho e depois preenche, evita
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount,
                            swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
  }

  // o pipeline nao consegue usar uma VkImage bruta, precisa de montar uma
  // ImageView interpretando e descrevendo como que acessa
  void createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());

    // pra cada imagem do swapchain, cria uma view
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = swapChainImages[i];
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = swapChainImageFormat;
      createInfo.components = {
          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
          VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;

      if (vkCreateImageView(device, &createInfo, nullptr,
                            &swapChainImageViews[i]) != VK_SUCCESS) {
        throw std::runtime_error("Erro ao criar image view");
      }
    }
  }

  void createRenderPass() {
    // Imagem da cor onde vai ser desenhado
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // imagem da profundidade onde vai ser desenhado
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    // a dependencia garante que não vai ter condicao de corrida na GPU tendo
    // certeza de que a placa de video vai esperar a imagem estar disponivel
    // antes de desenhar
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment,
                                                          depthAttachment};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro no render pass");
    }
  }

  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro pra criar o modulo de shader");
    }
    return shaderModule;
  }

  // criacao do pipeline grafico finalmente. Bastante longo por isso vou deixar
  // bastante anotação aqui. Primeiramente o pipeline grafico é basicamente a
  // linha de montagem que vai transformar os vertices em pixels coloridos, na
  // ordem: input assembly: le os vertices e agrupa em primitivas (formatos
  // geometricos basicos que a GPU renderiza na tela a partir dos vertices)
  // vertex shader: pra cada vertice, onde ele fica na tela
  // rasterizaçao: quais pixels a imagem cobre
  // fragment shader: para cada pixel, decide a cor
  // blending: mistura a cor nova com a que tava na imagem (como o alpha por
  // exemplo) imagem
  void createGraphicPipeline() {

    // le o vertex e o fragment shader
    auto vertShaderCode = readFile("shaders/triangle.vert.spv");
    auto fragShaderCode = readFile("shaders/triangle.frag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};

    // layout do buffer vertex, ou seja basicamente como os vertices vao chegar
    // no shader vertex (posicao, cor)
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    // o input assembly interpreta os dados de vertice e organiza nas primitivas
    // geometricas pra depois ser processados
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology =
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // como os vertices vao ser
                                             // agrupado, nesse caso pra cada 3
                                             // vertices el.e forma um triangulo
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    // como o pipeline vai mapear as coordenadas para os pixels
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    // o recorte que vai descartar os pixeels fora do retangulo definido, no
    // caso se tiver fora da swapchain (que ja é o tamanho da imagem inteira),
    // fodase
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent;

    // junta o viewport com o corte
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    // rasterizador que vai transformar os triangulos primitivos interpretados
    // no input assembly em pixels
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // VMQTG
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    // antialiasing
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    rasterizer.cullMode = VK_CULL_MODE_NONE;

    // como a cor de cada fragmento é misturada com a que já ta no framebuffer
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp =
        VK_COMPARE_OP_LESS; // ganha o menor z, ou seja, quem ta mais perto
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    // push constant, ta explicada la no shader vert
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4);

    // define quis recursos externos os shaders vao poder acessar.
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr,
                               &pipelineLayout) != VK_SUCCESS) {
      throw std::runtime_error("Erro no layout do pipeline");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.pDepthStencilState = &depthStencil;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                  nullptr, &graphicsPipeline) != VK_SUCCESS) {
      throw std::runtime_error("Erro ao criar o pipeline graficop");
    }

    // so precisa dos modulos dos shaders pra criar o pipeline grafico, depois
    // pode mandar pra pqp
    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
  }

  void createFramebuffers() {
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      std::array<VkImageView, 2> attachments = {swapChainImageViews[i],
                                                depthImageView};

      VkFramebufferCreateInfo framebufferInfo{};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = renderPass;
      framebufferInfo.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      if (vkCreateFramebuffer(device, &framebufferInfo, nullptr,
                              &swapChainFramebuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("Erro ao criar Framebuffer");
      }
    }
  }

  void createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // tag pra conseguir
                                                         // resetar os buffers
                                                         // de comando
                                                         // individualmente

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro na criacao do pool de comando");
    }
  }

  void createCommandBuffers() {
    commandBuffers.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro ao alocar os buffers na pool");
    }
  }

  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex,
                           const std::vector<glm::mat4> &mvps) {
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    VkBuffer vertexBuffers[] = {vertexBuffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

    // desenha mais de uma vez o mesmo vertez buffer mudando so a matriz
    for (const auto &mvp : mvps) {
      vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                         sizeof(glm::mat4), &mvp);
      uint32_t first =
          (sceneMode == 2) ? static_cast<uint32_t>(vertices.size()) : 0;
      uint32_t count = (sceneMode == 2)
                           ? PYRAMINX_VERTEX_COUNT
                           : static_cast<uint32_t>(vertices.size());
      vkCmdDraw(cmd, count, 1, first, 0);
    }

    vkCmdEndRenderPass(cmd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
      throw std::runtime_error("Erro na gravacao dos buffers de comando");
    }
  }

  void createSyncObjects() {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                          &imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateFence(device, &fenceInfo, nullptr, &inFlightFence) !=
            VK_SUCCESS) {
      throw std::runtime_error("Erro ao criar o objeto de sincronizacao");
    }

    // um semaforo pra cada imagem AUDIO NO GRUPO SOZINHHO EXPLICANDO PORQUE
    renderFinishedSemaphores.resize(swapChainImages.size());
    for (size_t i = 0; i < renderFinishedSemaphores.size(); i++) {
      if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                            &renderFinishedSemaphores[i]) != VK_SUCCESS) {
        throw std::runtime_error(
            "Erro tentando criar o semaforo de render pra imagem " +
            std::to_string(i));
      }
    }
  }

  // como eu ja disse la encima, o vkbuffer é um espaco linear de dados mas nao
  // tem memoria alocada, entao precisa encontrar exatamente um espaço de
  // memoria compativel com esse buffer
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }
    throw std::runtime_error(
        "Nao foi encontrado um tipo de memoria compativel");
  }

  void createVertexBuffer() {
    VkDeviceSize bufferSize =
        sizeof(Vertex) * (vertices.size() + PYRAMINX_VERTEX_COUNT);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro na criacao do vertex buffer");
    }

    // alocacao da memoria do buffer porque o buffer NAO ALOCA AUTOMATICAMENTE
    // tem que alocar na mao, ele é so um handle logico
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT // memoria visivel pra CPU tambem
                                            // (host)
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // quando a cpu escreve os
                                                     // dados nesse espaco de
                                                     // memoria, nao precisa
                                                     // chamar a funcao
                                                     // especifica pra garantir
                                                     // que as alteracoes da cpu
                                                     // em uma regia de memoria
                                                     // fiquem disponiveis para
                                                     // o dispositivo (gpu)

    if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("Nao deu pra alocar a mamoria do vertex");
    }

    vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

    // reescreve o pyraminx todo o frame durante todo o jogo
    vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &mappedVertices);
    memcpy(mappedVertices, vertices.data(), sizeof(Vertex) * vertices.size());
    pyraminx = makePyraminx();
  }

  // igual oitenta porcento das funcionalidades utilizadas ate entao tem que
  // veritifcar se a gpu tem suporte
  VkFormat findDepthFormat() {
    for (VkFormat f : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                       VK_FORMAT_D24_UNORM_S8_UINT}) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physicalDevice, f, &props);
      if (props.optimalTilingFeatures &
          VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
        return f;
      }
    }
    throw std::runtime_error(
        "Nao ha suporte pra nenhum formato de profundidade");
  }

  void createDepthResources() {
    depthFormat = findDepthFormat();

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent = {swapChainExtent.width, swapChainExtent.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &depthImage) != VK_SUCCESS) {
      throw std::runtime_error("Nao conseguiu criar a depth image");
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, depthImage, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(
        memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &depthImageMemory) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro ao alocar memoria pra depth image");
    }
    vkBindImageMemory(device, depthImage, depthImageMemory, 0);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(device, &viewInfo, nullptr, &depthImageView) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro ao criar a depth image view");
    }
  }

  // tudo daqui até o update input ta em audio
  void startMove(int v, bool tipOnly, bool prime) {
    if (animating)
      return;

    auto corners = tetraCorners();
    float R = glm::length(corners[v]);
    float r = std::sqrt(6.0f) / 12.0f * EDGE;
    float h = R + r;
    glm::vec3 u =
        glm::normalize(corners[v]);

    float limit = (tipOnly ? 0.33f : 0.67f) * h;
    moving.assign(pyraminx.size(), 0);
    for (size_t i = 0; i < pyraminx.size(); i++) {
      glm::vec3 c =
          (pyraminx[i].p[0] + pyraminx[i].p[1] + pyraminx[i].p[2]) / 3.0f;
      float depth = R - glm::dot(c, u);
      moving[i] = depth < limit;
    }

    moveAxis = u;
    moveAngle = 0.0f;
    moveTarget = glm::radians(
        prime ? 120.0f : -120.0f);
    animating = true;
  }

  void updatePyraminx(float dt) {
    if (!animating)
      return;

    float step = glm::radians(360.0f) * dt;
    moveAngle += (moveTarget > 0 ? step : -step);

    if (std::abs(moveAngle) >= std::abs(moveTarget)) {
      glm::quat q = glm::angleAxis(moveTarget, moveAxis);
      for (size_t i = 0; i < pyraminx.size(); i++) {
        if (!moving[i])
          continue;
        for (int k = 0; k < 3; k++)
          pyraminx[i].p[k] = q * pyraminx[i].p[k];
      }
      animating = false;
    }
  }

  void uploadPyraminx() {
    glm::quat q =
        animating ? glm::angleAxis(moveAngle, moveAxis) : glm::quat(1, 0, 0, 0);
    const glm::vec3 black(0.05f);
    std::vector<Vertex> out;
    out.reserve(PYRAMINX_VERTEX_COUNT);

    for (size_t i = 0; i < pyraminx.size(); i++) {
      glm::vec3 P[3];
      for (int k = 0; k < 3; k++) {
        P[k] =
            (animating && moving[i]) ? q * pyraminx[i].p[k] : pyraminx[i].p[k];
      }
      glm::vec3 cen = (P[0] + P[1] + P[2]) / 3.0f;

      for (int k = 0; k < 3; k++)
        out.push_back({P[k] * 0.96f, black});

      for (int k = 0; k < 3; k++)
        out.push_back({cen + (P[k] - cen) * 0.88f, pyraminx[i].color});
    }
    memcpy(static_cast<Vertex *>(mappedVertices) + vertices.size(), out.data(),
           out.size() * sizeof(Vertex));
  }

  void updateInput(float dt) {
    float moveSpeed = 1.0f;
    float rotSpeed = glm::radians(90.0f);

    if (glfwGetKey(window, GLFW_KEY_F1) == GLFW_PRESS)
      sceneMode = 0;
    if (glfwGetKey(window, GLFW_KEY_F2) == GLFW_PRESS)
      sceneMode = 1;
    if (glfwGetKey(window, GLFW_KEY_F3) == GLFW_PRESS)
      sceneMode = 2;

    // setinha: movimenta no eixo x e y
    // w e s: movimenta no eixo z
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
      position.x += moveSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
      position.x -= moveSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
      position.y += moveSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
      position.y -= moveSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      position.z -= moveSpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
      position.z += moveSpeed * dt;

    // I e K giram no eixo x, J e L no eixo y, U e O no eixo z
    // rotacionar: no eixo x, no eixo y, no eixo z
    auto rotate = [&](glm::vec3 axis, float dir) {
      orientation = glm::normalize(glm::angleAxis(dir * rotSpeed * dt, axis) *
                                   orientation);
    };
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
      rotate({1, 0, 0}, 1.0f);
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
      rotate({1, 0, 0}, -1.0f);
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
      rotate({0, 1, 0}, 1.0f);
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
      rotate({0, 1, 0}, -1.0f);
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
      rotate({0, 0, 1}, 1.0f);
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
      rotate({0, 0, 1}, -1.0f);

    // rotacionar: em torno de um eixo qualquer(x,y,z)
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
      rotate(glm::normalize(glm::vec3(1, 1, 0)), 1.0f);

    // rotacionar em função dos 4 eixos que passam por cada um de seus vertices
    auto corners = tetraCorners();
    float dir =
        (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? -1.0f : 1.0f;
    for (int i = 0; i < 4; i++) {
      if (sceneMode != 2 && glfwGetKey(window, GLFW_KEY_1 + i) == GLFW_PRESS) {
        rotate(glm::normalize(orientation * corners[i]), dir);
      }
    }

    if (sceneMode == 2) {
      // 1 ao 4 mexe a camada grande
      // 5 a 8 gira as pontas
      // segura shift gira ao inverso
      bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
      for (int i = 0; i < 8; i++) {
        bool down = glfwGetKey(window, GLFW_KEY_1 + i) == GLFW_PRESS;
        if (down && !keyWasDown[i])
          startMove(i % 4, i >= 4, shift);
        keyWasDown[i] = down;
      }
    }
    updatePyraminx(dt);
  }
  // quando voce vai girar o objeto precisa obviamente tambem calcular como
  // que o objeto vai ser posicionado/girado(model), como a camera vai ser
  // posicionada/direcionada (view) e como tudo isso vai ser projetado
  // (projection), ou seja essa funcao monta a matrix mvp
  std::vector<glm::mat4> buildMVPs() {
    // Transformação do grupo todo (as teclas de translação/rotação mexem em
    // todos juntos)
    glm::mat4 group =
        glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(orientation);

    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f),
                                 glm::vec3(0.0f, 1.0f, 0.0f));
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 20.0f);
    proj[1][1] *= -1;

    std::vector<glm::mat4> mvps;
    for (const auto &local : buildLocalTransforms()) {
      mvps.push_back(proj * view * group * local);
    }
    return mvps;
  }

  void drawFrame() {
    // espera a placa de video terminar de processar e depois reiniciar pra
    // poder sinalizar o proximo
    vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &inFlightFence);

    // pede a nova imagem pra swapchain
    uint32_t imageIndex;
    vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
                          imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (sceneMode == 2)
      uploadPyraminx();

    // grava os comandos e constroi as matrizes mvp
    recordCommandBuffer(commandBuffers[imageIndex], imageIndex, buildMVPs());

    // configura as infos que vai mandar pra fila de processamento
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // aguarda conseguir a imagem da swapchain antes de liberar
    VkSemaphore waitSemaphores[] = {imageAvailableSemaphore};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    // define o buffer de comando com os comandos de renderizar o frame
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    // sinaliza quando os comandos da fila terminarem
    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    // envia os comandos pra graphic queue e conecta com a fence dessa rodada
    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) !=
        VK_SUCCESS) {
      throw std::runtime_error("Erro na hora de enviar o buffer de comando");
    }

    // config da apresentacao da imagem
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    // escolhe a imagem e a swapchain que vai apresentar
    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    // pedem pra fila de apresnetar... apresentar.
    vkQueuePresentKHR(presentQueue, &presentInfo);
  }

  void mainLoop() {
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
      double now = glfwGetTime();
      float dt = static_cast<float>(now - lastTime);
      lastTime = now;

      glfwPollEvents();
      updateInput(dt);
      drawFrame();
    }
    // esperar a placa processar aboslutamente tudo antes de dar o cleanup
    vkDeviceWaitIdle(device);
  }

  // Vulkan nao tem garbage collector (bom que já estou acostumado tendo
  // trabalhado bastante com ADVPL e TLPP)
  void cleanup() {
    // tem que destruir na ordem INVERSA que foi criado se não crasha tudo
    for (auto semaphore : renderFinishedSemaphores) {
      vkDestroySemaphore(device, semaphore, nullptr);
    }
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkUnmapMemory(device, vertexBufferMemory);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (auto framebuffer : swapChainFramebuffers) {
      vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);

    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapChainImageViews) {
      vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapChain, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
  }
};

int main() {

  PyraminxApp app;

  try {

    app.run();
  } catch (const std::exception &e) {

    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}