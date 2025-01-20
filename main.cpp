#ifdef WIN32
#include <SDL.h>
#undef main
#else
#include <SDL2/SDL.h>
#endif

#include <GL/glew.h>

#include <string_view>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <cmath>

#define GLM_FORCE_SWIZZLE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtx/string_cast.hpp>

#include "obj_parser.hpp"
#include "stb_image.h"


// ===== ОБЩИЕ ФУНКЦИИ =====
std::string to_string(std::string_view str)
{
    return std::string(str.begin(), str.end());
}

void sdl2_fail(std::string_view message)
{
    throw std::runtime_error(to_string(message) + SDL_GetError());
}

void glew_fail(std::string_view message, GLenum error)
{
    throw std::runtime_error(to_string(message) + reinterpret_cast<const char *>(glewGetErrorString(error)));
}

GLuint create_shader(GLenum type, const char * source)
{
    GLuint result = glCreateShader(type);
    glShaderSource(result, 1, &source, nullptr);
    glCompileShader(result);
    GLint status;
    glGetShaderiv(result, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetShaderiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetShaderInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Shader compilation failed: " + info_log);
    }
    return result;
}

GLuint create_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint result = glCreateProgram();
    glAttachShader(result, vertex_shader);
    glAttachShader(result, fragment_shader);
    glLinkProgram(result);

    GLint status;
    glGetProgramiv(result, GL_LINK_STATUS, &status);
    if (status != GL_TRUE)
    {
        GLint info_log_length;
        glGetProgramiv(result, GL_INFO_LOG_LENGTH, &info_log_length);
        std::string info_log(info_log_length, '\0');
        glGetProgramInfoLog(result, info_log.size(), nullptr, info_log.data());
        throw std::runtime_error("Program linkage failed: " + info_log);
    }

    return result;
}


// ===== ШЕЙДЕРЫ ДЛЯ РИСОВАНИЯ ПЛАНЕТЫ (ЗЕМЛИ)=====
const char vertex_shader_source[] =
R"(#version 330 core

uniform mat4 model;  // матрица, переводящая из системы кооррдинат модели (самой планеты) в мировую систему координат (самой сцены) -> когда один объект (как у нас), то чаще всего это тождественная (те единичная) матрица
uniform mat4 view;  // матрицы, отвечающие за параметры камеры: view, projection (они нужны для перспективной проекции...)
uniform mat4 projection;  
uniform sampler2D height_texture;  // карта высот (текстура, где значения пикселей в gray-scale характеризуют высоту данной точки на поверхности Земли: 0 - уровень моря, 1 - наивысшая точка = Эверест)

layout (location = 0) in vec3 in_position;  // аттрибуты вершин сферы, которая лежит в основе модели Земли: 3d-координата данной (от которой запускается шейдер) вершины,
layout (location = 1) in vec3 in_tangent;  // тангент вектор в ней,
layout (location = 2) in vec3 in_normal;  // нормаль к поверхности сферы в данной вершине
layout (location = 3) in vec2 in_texcoord;  // текстурные координаты (те координаты в текстурах (эти координаты нормированы и находятся в [0,1]), откуда читать значения для этой вершинки)

out vec3 position;  // аналогичные переменные, через которые передадим дальше значения в фрагментный шейдер
out vec3 tangent;
out vec3 normal;
out vec2 texcoord;

void main()
{
    texcoord = vec2(1.0 - in_texcoord.x, 1.0 - in_texcoord.y);  // x и y текстурные координаты - значения от 0 до 1 -> чтобы картинка выглядела не перевёрнуто, нужно эмпирически понять, какая координата нужна: x или 1-x и тд
    
    vec3 direction = normalize(in_position);  // единичный вектор направления в точку с координатами position из (0, 0, 0) = центр модели = центр Земли
    float delta = 0.05 * texture(height_texture, texcoord).r;  // сдвиг согласно карте высот (константу 0.05 подбираем эмпирически, чтобы выглядело заметно... - это значение = свдиг для самых высоких гор, имеющим на карте значение = 1 -> высота Эвереста = 0.05;)
                                                               // кстати, у меня радиус сферы в данном проекте = 1, а в реальности = 6400км -> в моей модели высота Эвереста = 0.05/1 * 6400 = 320 км (а в реальности всего 8 км... но так было специально увеличено, чтобы горы были заметны)
    position = (model * vec4(in_position + delta * direction, 1.0)).xyz;  // получаем 3d-координату точки в пространстве (после применения model)... не забываем сдвинуть точку на delta в направлении direction (этот сдвиг даёт выпуклость гор, так как direction сонаправлен с радиусом, те от центра Земли смотрит)
    gl_Position = projection * view * vec4(position, 1.0);  // устанавливаем координаты точки, которую в итоге шейдер нарисует
    tangent = mat3(model) * in_tangent;  // пробрасываем далее значения
    normal = mat3(model) * in_normal;
}
)";

// по-хорошему, прежде чем переходить к фрагментному шейдеру, нужно написать геометрический шейдер, который будет набирать по три вершины (= целый треугольник),
// а затем считать для каждой вершины нормаль = перепендикуляр к этому треугольнику (те во всем треугольнике нормаль будет константна = нормаль ко всему трекгольнику) - заметим, что
// в геом шейдер уже приходят индивидульные вершины (если треугольники соприкасаются в одной и той же вершине, то существует несколько копий этой вершины - по одной для каждого треугольника)
// (сама процедура с геом шейдером нужна, так как от вытягивания гор, нормали, которые были ранее для сферы, стали не очень корректны... -> но ничего страшного, мы сделаем иначе:
// с помощью карты нормалей сделаем normal mapping, получив не приближенные нормали через геом шейдер, а более правдободобные из карты нормалей)

const char fragment_shader_source[] =
R"(#version 330 core

uniform vec3 camera_position;  // положение камеры в сцене

uniform sampler2D albedo_texture;  // текстура с собственным цветом объекта (= планеты Земля)
uniform sampler2D normal_texture;  // текстура с нормалями к поверхности (карта нормалей)
uniform sampler2D glossiness_texture;  // текстура со значениями glossiness (некоторый множитель из brdf, которая опеределяет, насколько сильно отражает объект)
uniform sampler2D night_texture;  // собственное свечение Земли ночью

in vec3 position;  // положение данного (для которого запущен шейдер) пикселя в пространстве
in vec2 texcoord;  // и другие параметры, пришедшие из предыдущего шейдера
in vec3 tangent;
in vec3 normal;

uniform vec3 sun_direction;  // направление на Солнце
layout (location = 0) out vec4 out_color;  // выходной цвет, который мы хотим высчитать, чтобы его назначили пикселю

vec3 real_normal;  // настоящая нормаль (она будет подправлять исходную нормаль normal сферы, учитывая карту нормалей)

// Функция, которая вычисляет коэффициент (на который свет Солнца или другого источника домножается, и полученное значение добавляется к цвету пикселя) для diffuse-освещения:
vec3 diffuse(vec3 direction) {
    vec3 albedo = texture(albedo_texture, texcoord).rgb;  // из текстуры читаем значение альбедо данного пикселя
    return albedo * max(0.0, dot(real_normal, direction));  
}

// Аналогично функция, получающая specular-коэффициент (direction - направление на источник света)
vec3 specular(vec3 direction) {
    float cosine = dot(real_normal, direction);  // косинус угла падения лучей от Солнца на Землю (данную её точку)
    vec3 reflected = 2.0 * real_normal * cosine - direction; // отражённый в данной точке луч от исчтоника,
    vec3 view_direction = normalize(camera_position - position);  // направление из этой точки в камеру...
    vec3 glossiness = texture(glossiness_texture, texcoord).rgb;  // не забываем счтать коэффициент
    vec3 albedo = texture(albedo_texture, texcoord).rgb;
    return glossiness * albedo * pow(max(0.0, dot(reflected, view_direction)), 4.5);  // 4.5 - насколько сильно отражает (чем больше, тем зеркальнее)
}

// Итоговый коэффициент - сумма двух (ещё есть ambient-освещение, но в космосе его нет, тк ambient обычно моделирет атмосферу) - это и есть модель Фонга
vec3 phong(vec3 direction) {
    return diffuse(direction) + specular(direction);
}

// Функция для tonemap цвета:
vec3 tonemap(vec3 x)
{
	float a = 2.51;
	float b = 0.03;
	float c = 2.43;
	float d = 0.59;
	float e = 0.14;
	return clamp((x*(a*x+b))/(x*(c*x+d)+e), vec3(0.0), vec3(1.0));
}

void main()
{   
    // Считаем реальную нормаль (ormal mapping) из карты нормалей и тангент-битангент векторов:
    vec3 bitangent = cross(tangent, normal);
    mat3 tbn = mat3(tangent, bitangent, normal);
    real_normal = normalize(tbn * ((texture(normal_texture, texcoord).xyz * 2.0 - vec3(1.0)) * vec3(-1, -1, 1.0/10.0)));  // пока мы не домножили на tbn, мы работаем в локальной системе координат (когда x,y-вдоль поверзности Земли, а z - нормаль к поверхности сферы) -> тут эмпирически стало понятно, что x и y нужно инвертировать (домножаем на -1); также, учитывая, что горы мы вытянули (в 40 раз: вместо 8км Эвереста целых 320км!), также скалируем и ось z (чтобы нормали стали сильнее соответсвенно вытягиванию гор)... константу 1/10 оже на глаз берем...
    
    // Считаем наконец цвет:
    vec3 sun_color = 3 * vec3(1.0, 1.0, 1.0);  // в космосе Солнце белое - r=g=b=1.0; также умножаем на 3, чтобы поярче было
    float cosine = dot(real_normal, sun_direction);  // косинус угла падения Солнечного света на Землю

    float alpha = smoothstep(-0.1, 0.1, cosine);  // считаем коэффициент для будущей интерполяции денвой стороны Земли и ночной: если косинус слишком мал (< -0.1) = однозначно ночь, то smoothstep вернёт 0
                                                  // если > 0.1 = день, то значение alpha = 1    (а для значений cosine от -0.1 до 0.1 будет интерполяция -> alpha будет плавно от 0 до 1....)
    vec3 day_color = phong(sun_direction) * sun_color;  // (если день) считаем цвет точки (= пикселя) на Земле днём - просто освещение от Солнца
    vec3 night_color = 4 * texture(night_texture, texcoord).rgb;  // (если ночь) сама ХЗемля светится огнями городов - читаем из текстуры и домножаем (на 4), чтобы ярче стало
    vec3 color = mix(night_color, day_color, alpha);  // смешиваем для гладкой интерпляции: если alpha = 0, то остается только night_color (будет ночь), если alpha=1, nо только дневное освещение... а между, когда alpha от 0 до 1, делаем плавный переход
    color = tonemap(color) / tonemap(vec3(11.2));  // делаем tonemap
    color = pow(color, vec3(1.0/2.2));  // делаем гамма-коррекцию
    out_color = vec4(color, 1.0); 
    //out_color = vec4(cosine, 0.0, 0.0, 1.0);  // для отладки можно только косинус угла падения вывести - он характеризует нормали...
}
)";


// ===== ШЕЙДЕРЫ ДЛЯ РИСОВАНИЯ ENVIRONMENT MAP (фон из звёзд) =====
const char env_vertex_source[] = 
R"(#version 330 core

const vec2 VERTICES[6] = vec2[6](  // захардкоженные вершины двух треугольников, образующих прямоугольник на весь экран (координаты +-1 по x и y)
    vec2(-1.0, -1.0),  // три вершины (в порядке против часовой стрелки) первого треугольника (левый верхний)
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    
    vec2(-1.0, -1.0),  // второй треугольник
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
);

uniform mat4 view;  // матрицы view и projection те же самые, что и для сферы (те же, что изначально в коде были) - так как они просто камеру характеризуют
uniform mat4 projection;

out vec3 position;  // позиция в пространстве точки, изображение которой является вершиной треугольников - это интерполируется и передаётся во фрагментный шейдер далее

void main() {
    vec2 vertex = VERTICES[gl_VertexID];  // просто по индексу берем новую точку из массива
    gl_Position = vec4(vertex, 0.0, 1.0);  // вершина, которую нужно нарисовать
    vec4 ndc = vec4(vertex, 0.0, 1.0);
    vec4 clip_space = inverse(projection * view) * ndc;  // прообраз вершины в пространстве
    position = clip_space.xyz / clip_space.w;
}
)";

const char env_fragment_source[] =
R"(#version 330 core

uniform sampler2D environment_texture;  // текстура, которую рисуем на фоне
uniform vec3 camera_position;  // положение камеры

in vec3 position;  // координаты точки в пространстве из шейдера ранее

layout (location = 0) out vec4 out_color;

const float PI = 3.141592653589793;

void main() {
    vec3 dir = position - camera_position;  // !!! важно - нас тут интересует направдение из камеры в точку (а не как ранее наоборот)
    float theta = atan(dir.z, dir.x) / PI * 0.5 + 0.5;  // аналогично предыдущему выисляем координаты (а именно - широту и долготу точки position относительно камеры)
    float phi = -atan(dir.y, length(dir.xz)) / PI + 0.5;
    theta = 1 - theta;  // из практики понимаем, что изображение перевёрнуто... -> инвертируем
    out_color = vec4(texture(environment_texture, vec2(theta, phi)).rgb, 1.0);  // используем для получения значения из текстуры
}
)";


// ===== ГЕНЕРАЦИЯ ВЕРШИН СФЕРЫ ===
struct vertex  // каждая вершина сферы имеет несколько полей: 3d-координату, тангент вектор, нормаль, текстурные коордианты
{
    glm::vec3 position;
    glm::vec3 tangent;
    glm::vec3 normal;
    glm::vec2 texcoords;
};

std::pair<std::vector<vertex>, std::vector<std::uint32_t>> generate_sphere(float radius, int quality)  // как-то генерируем вершины и индексы сферы радиуса radius (quilty - чем больше, тем более гладкая сфера и более правильная будущая Земля)
{
    std::vector<vertex> vertices;

    for (int latitude = -quality; latitude <= quality; ++latitude)
    {
        for (int longitude = 0; longitude <= 4 * quality; ++longitude)
        {
            float lat = (latitude * glm::pi<float>()) / (2.f * quality);
            float lon = (longitude * glm::pi<float>()) / (2.f * quality);

            auto & vertex = vertices.emplace_back();
            vertex.normal = {std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon)};
            vertex.position = vertex.normal * radius;
            vertex.tangent = {-std::cos(lat) * std::sin(lon), 0.f, std::cos(lat) * std::cos(lon)};
            vertex.texcoords.x = (longitude * 1.f) / (4.f * quality);
            vertex.texcoords.y = (latitude * 1.f) / (2.f * quality) + 0.5f;
        }
    }

    std::vector<std::uint32_t> indices;

    for (int latitude = 0; latitude < 2 * quality; ++latitude)
    {
        for (int longitude = 0; longitude < 4 * quality; ++longitude)
        {
            std::uint32_t i0 = (latitude + 0) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i1 = (latitude + 1) * (4 * quality + 1) + (longitude + 0);
            std::uint32_t i2 = (latitude + 0) * (4 * quality + 1) + (longitude + 1);
            std::uint32_t i3 = (latitude + 1) * (4 * quality + 1) + (longitude + 1);

            indices.insert(indices.end(), {i0, i1, i2, i2, i1, i3});
        }
    }

    return {std::move(vertices), std::move(indices)};
}


// Функция для загрузки текстуры из файла-картинки по пути path сразу на gpu (графическую карту):
GLuint load_texture(std::string const & path, bool is_albedo=false)
{
    int width, height, channels;
    auto pixels = stbi_load(path.data(), &width, &height, &channels, 4);  // читаем картинку как 4-ёх канальную (r,g,b - цветовые каналы и ещё a-канал прозрачности) - у нас все в opengl такое

    GLuint result;
    glGenTextures(1, &result);
    glBindTexture(GL_TEXTURE_2D, result);
    glTexImage2D(GL_TEXTURE_2D, 0, is_albedo ? GL_SRGB8 : GL_RGB8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);  // не забываем, что albedo-текстуры грузим как sRGB! (для этого есть флаг is_albedo)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);  // не забываем сгенерировать уровни mipmap
    stbi_image_free(pixels);

    return result;
}



int main() try
{
    // === ОБЩИЕ НАСТРОЙКИ РИСОВАНИЯ ===
    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdl2_fail("SDL_Init: ");

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window * window = SDL_CreateWindow("Graphics course practice 5",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800, 600,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED);

    if (!window)
        sdl2_fail("SDL_CreateWindow: ");

    int width, height;
    SDL_GetWindowSize(window, &width, &height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
        sdl2_fail("SDL_GL_CreateContext: ");

    if (auto result = glewInit(); result != GLEW_NO_ERROR)
        glew_fail("glewInit: ", result);

    if (!GLEW_VERSION_3_3)
        throw std::runtime_error("OpenGL 3.3 is not supported");

    glClearColor(0.0f, 0.0f, 0.f, 0.f);  // цвет фона - чёрный (Земля в космосе)

    // === ПОЛУЧАЕМ ШЕЙДЕРНУЮ ПРОГРАММУ И ПОЛУЧАЕМ УКАЗАТЕЛИ НА ЕЁ ПЕРЕМЕННЫЕ (uniform) ===
    auto vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source);
    auto fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source);
    auto program = create_program(vertex_shader, fragment_shader);

    GLuint model_location = glGetUniformLocation(program, "model");
    GLuint view_location = glGetUniformLocation(program, "view");
    GLuint projection_location = glGetUniformLocation(program, "projection");
    GLuint camera_position_location = glGetUniformLocation(program, "camera_position");
    GLuint sun_direction_location = glGetUniformLocation(program, "sun_direction");
    GLuint albedo_texture_location = glGetUniformLocation(program, "albedo_texture");
    GLuint normal_texture_location = glGetUniformLocation(program, "normal_texture");
    GLuint glossiness_texture_location = glGetUniformLocation(program, "glossiness_texture");
    GLuint night_texture_location = glGetUniformLocation(program, "night_texture");
    GLuint height_texture_location = glGetUniformLocation(program, "height_texture");
    
    // === АНАЛОГИЧНО ДЛЯ ENVIRONMENT ===
    auto env_vertex_shader = create_shader(GL_VERTEX_SHADER, env_vertex_source);
    auto env_fragment_shader = create_shader(GL_FRAGMENT_SHADER, env_fragment_source);
    auto env_program = create_program(env_vertex_shader, env_fragment_shader);
    GLuint env_view_location = glGetUniformLocation(env_program, "view");
    GLuint env_projection_location = glGetUniformLocation(env_program, "projection");
    GLuint environment_texture_location = glGetUniformLocation(env_program, "environment_texture");
    GLuint env_camera_position_location = glGetUniformLocation(env_program, "camera_position");
    GLuint empty_vao;
    glGenVertexArrays(1, &empty_vao);  // фиктивный vao для рисования прямоугольника environament

    // === ПРОЛУЧАЕМ БУФЕРЫ ДЛЯ СФЕРЫ ===
    GLuint sphere_vao, sphere_vbo, sphere_ebo;
    glGenVertexArrays(1, &sphere_vao);
    glBindVertexArray(sphere_vao);
    glGenBuffers(1, &sphere_vbo);
    glGenBuffers(1, &sphere_ebo);
    GLuint sphere_index_count;
    {
        auto [vertices, indices] = generate_sphere(1.f, 54);

        glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vertices[0]), vertices.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]), indices.data(), GL_STATIC_DRAW);

        sphere_index_count = indices.size();
    }
    glEnableVertexAttribArray(0);  // устанвливаем аттрибуты вершин 
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, position));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, tangent));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, normal));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(vertex), (void *)offsetof(vertex, texcoords));

    // === ЗАГРУЖАЕМ ТЕКСТУРЫ ===
    std::string project_root = PROJECT_ROOT;
    GLuint albedo_texture = load_texture(project_root + "/textures/albedo.jpg", true);
    GLuint normal_texture = load_texture(project_root + "/textures/normal.jpg");
    GLuint glossiness_texture = load_texture(project_root + "/textures/glossiness.jpg");
    GLuint night_texture = load_texture(project_root + "/textures/night.jpg", true);
    GLuint height_texture = load_texture(project_root + "/textures/heights.jpg");
    GLuint env_texture = load_texture(project_root + "/textures/8k_stars_milky_way.jpg");

    // === ОСНВНОЙ ЦИКЛ РИСОВАНИЯ ===
    auto last_frame_start = std::chrono::high_resolution_clock::now();
    float time = 0.f;
    std::map<SDL_Keycode, bool> button_down;  // словаоь для сохранения нажатых кнопок
    float view_elevation = glm::radians(30.f);  // изначальный угол наклона камеры = 30 градусов по широте (камера у нас всегда смотрит в центр сцены = (0,0,0) и поворачивается по широте/долготе вокруг это точки... очень удобно - так Землю можем со всех ракурсов рассмотреть)
    float view_azimuth = 0.f;  // по долготе (азимуту) - 0 градусов
    float camera_distance = 2.f;  // расстояние от (0,0,0) до камеры

    bool paused = false;
    bool running = true;
    while (running)
    {
        // === Обновляем нажатия кнопок и соответсвенные изменения положения камер ===
        for (SDL_Event event; SDL_PollEvent(&event);) switch (event.type) {
            case SDL_QUIT:
                running = false;
                break;
            case SDL_WINDOWEVENT: switch (event.window.event)
                {
                case SDL_WINDOWEVENT_RESIZED:
                    width = event.window.data1;
                    height = event.window.data2;
                    glViewport(0, 0, width, height);
                    break;
                }
                break;
            case SDL_KEYDOWN:
                button_down[event.key.keysym.sym] = true;
                if (event.key.keysym.sym == SDLK_SPACE)
                    paused = !paused;
                break;
            case SDL_KEYUP:
                button_down[event.key.keysym.sym] = false;
                break;
        }

        if (!running)
            break;

        auto now = std::chrono::high_resolution_clock::now();  // обновляем время, прошедшее с предыдущего кадра
        float dt = std::chrono::duration_cast<std::chrono::duration<float>>(now - last_frame_start).count();
        last_frame_start = now;

        if (!paused)
            time += dt;

        if (button_down[SDLK_UP])  // управление камерой: W-A-S-D клавиши поворачивают камеру по широте/долготе вокруг (0,0,0), а стрелочк вверх/вниз приближают/удаляют
            camera_distance -= 4.f * dt;
        if (button_down[SDLK_DOWN])
            camera_distance += 4.f * dt;

        if (button_down[SDLK_a])
            view_azimuth -= 2.f * dt;
        if (button_down[SDLK_d])
            view_azimuth += 2.f * dt;

        if (button_down[SDLK_w])
            view_elevation -= 2.f * dt;
        if (button_down[SDLK_s])
            view_elevation += 2.f * dt;

        float near = 0.1f;  // параметры near/far для камеры
        float far = 100.f;
        float top = near;
        float right = (top * width) / height;

        // === Настройка параметров камеры и света ===
        glm::mat4 model = glm::mat4(1.f);  // матрица model единичная (центр Земли = центр сцены = 0,0,0)

        glm::mat4 view(1.f);  // матрицы камеры получаем
        view = glm::translate(view, {0.f, 0.f, -camera_distance});
        view = glm::rotate(view, view_elevation, {1.f, 0.f, 0.f});
        view = glm::rotate(view, view_azimuth, {0.f, 1.f, 0.f});  

        glm::mat4 projection = glm::mat4(1.f);
        projection = glm::perspective(glm::pi<float>() / 2.f, (1.f * width) / height, near, far);

        glm::vec3 camera_position = (glm::inverse(view) * glm::vec4(0.f, 0.f, 0.f, 1.f)).xyz();

        glm::vec3 sun_direction;  // получаем направление на Солнце
        float sun_speed = 0.25;
        glm::mat4 rotation_matrix = glm::rotate(glm::mat4(1.0f), glm::radians(23.5f), glm::vec3(1.0f, 0.0f, 0.0f));  // матрица поворота на 23.5 градуса вокруг оси x
        sun_direction = glm::normalize(glm::vec3(rotation_matrix * glm::vec4(std::sin(time * sun_speed), 0.f, std::cos(time * sun_speed), 1.0)));  // направление на Солнце (ось вращения Земли наклонена к плоскости эклиптики на 23.5 градуса -> со стороны Земли выглядит так, что Солнце вращается под таким углом вокруг Земли -> поворачиваем солнце в плоскости xz и наклоняем затем на 23.5 градуса)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);

        // === Рисуем фон из звёзд ===
        glUseProgram(env_program);  // не забываем обязательно включить нашу с шейдерами программу
        glUniformMatrix4fv(env_view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(env_projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(env_camera_position_location, 1, reinterpret_cast<float *>(&camera_position));
        glActiveTexture(GL_TEXTURE0); 
        glBindTexture(GL_TEXTURE_2D, env_texture);
        glUniform1i(environment_texture_location, 0);  // используем текстуру в шейдере через 0-ой texture-uni
        glDisable(GL_DEPTH_TEST);  // рисуем фон БЕЗ теста глубины (как в лекциях написано)
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);  // рисуем треугольники их 6 вершин, начиная с 0-ой (то есть индекс gl_VertexID, который передастся в вершинный шейдер будет от 0 до 5)
        glEnable(GL_DEPTH_TEST);  // обратно включаем

        // === Рисуем саму Землю ===
        glUseProgram(program);  // обязательно включаем использование программы перед установкой параметров
        glActiveTexture(GL_TEXTURE0);  // по пордяку в каждом из texture unit (с номерами 0-4) делаем текущей свою текстуру и передаём информацию об этом в шейдер (тогда шейдер значет, из какой именно текстурки читать)
        glBindTexture(GL_TEXTURE_2D, albedo_texture);
        glUniform1i(albedo_texture_location, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normal_texture);
        glUniform1i(normal_texture_location, 1);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, glossiness_texture);
        glUniform1i(glossiness_texture_location, 2);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D,  night_texture);
        glUniform1i(night_texture_location, 3);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D,  height_texture);
        glUniform1i(height_texture_location, 4);

        glUniformMatrix4fv(model_location, 1, GL_FALSE, reinterpret_cast<float *>(&model));  // устанавливаем оставшиеся параметры
        glUniformMatrix4fv(view_location, 1, GL_FALSE, reinterpret_cast<float *>(&view));
        glUniformMatrix4fv(projection_location, 1, GL_FALSE, reinterpret_cast<float *>(&projection));
        glUniform3fv(camera_position_location, 1, (float *) (&camera_position));
        glUniform3f(sun_direction_location, sun_direction.x, sun_direction.y, sun_direction.z);

        glBindVertexArray(sphere_vao);  // наконец, рисуем
        glDrawElements(GL_TRIANGLES, sphere_index_count, GL_UNSIGNED_INT, nullptr);

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
}
catch (std::exception const & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
