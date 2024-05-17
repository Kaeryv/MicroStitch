#include <vector>
#include <fstream>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <set>


#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#undef  RAYGUI_IMPLEMENTATION            
#define GUI_FILE_DIALOG_IMPLEMENTATION
#include "gui_file_dialog.h"

#include "quickshift.h"
#define STB_DS_IMPLEMENTATION
#include "stb_ds.h"
#include "graphs.h"
#include "img_manipulation.h"
#include "core.h"

#define STITCH_GUI_TICK_HEIGHT 0.05f
#define STITCH_GUI_TICK_THICKNESS  0.02f
#define STITCH_GUI_AXIS_THICKNESS  0.02f

using json = nlohmann::json;

void
DrawGuidingLines(Color color, Camera2D cam) {
  float tick_height = STITCH_GUI_TICK_HEIGHT;
  DrawLineEx({-1000.f, 0.f}, {1000.f, 0.f}, STITCH_GUI_AXIS_THICKNESS, color);
  DrawLineEx({0.f, -1000.f}, {0.f, 1000.f}, STITCH_GUI_AXIS_THICKNESS, color);
  
  for (int i = cam.target.x-6; i < cam.target.x+6; i++) {
    DrawLineEx({(float) i, -tick_height}, {(float)i, tick_height}, STITCH_GUI_TICK_THICKNESS, color);
  }
  for (int i = cam.target.y-6; i < cam.target.y+6; i++) {
    DrawLineEx({-tick_height, (float) i}, {tick_height, (float)i}, STITCH_GUI_TICK_THICKNESS, color);
  }
}

struct AdvancedWindowState {
    bool windowOpened = true;
    Rectangle position = { 0.0, 0.0, 400, 300};
    const char *title;
    bool dragged = false;
};
Image MarkBoundaries(Image background, std::size_t* segmentation, Color color) {
  int width = background.width;
  int height = background.height;
  Image marked_bg = ImageCopy(background);
  #define seg_at(x, y) (((std::size_t*)segmentation)[(x)*width+(y)])
  #define bg_at(x, y) (((Color*)marked_bg.data)+(x)*width+(y))
  #pragma omp parallel for shared(background, segmentation, marked_bg) firstprivate(height,width)
  for (int i = 1; i < height-1; i ++){
        for (int j = 1; j < width-1; j ++) {
          bool boundary = seg_at(i, j) != seg_at(i, j+1);
          boundary |= seg_at(i, j-1) != seg_at(i, j);
          boundary |= seg_at(i+1, j) != seg_at(i, j);
          boundary |= seg_at(i-1, j) != seg_at(i, j);
          *bg_at(i,j) = boundary ? color : *bg_at(i,j);
        }
  }
  #undef seg_at
  #undef bg_at
  return marked_bg;
}

bool 
GuiAdvancedWindow(AdvancedWindowState *state) {
    if(state->windowOpened) {
        state->windowOpened = !GuiWindowBox(state->position, state->title);
    }
    // Do we move the window around ?
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        auto mp = GetMousePosition();
        Rectangle statusbarRect = state->position;
        statusbarRect.height = RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT;
        if (CheckCollisionPointRec(mp, statusbarRect)) {
            state->dragged = true;
        }
    } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        state->dragged = false;
    }

    if (state->dragged) {
        auto md = GetMouseDelta();
        state->position.x += md.x; 
        state->position.y += md.y;
    }
    return state->windowOpened;
}

struct PhiMap {
  PhiMap(float _x, float _y, float _w, float _h, int _r, Texture2D _tex, bool _s, bool _mb, int _fn) {
    x = _x;
    y = _y;
    w = _w;
    h = _h;
    rotation_deg = _r;
    tex = _tex;
    selected = _s;
    mouse_bound = _mb;
    fileno = _fn;
  }
  float x, y;
  float w, h;
  float rotation_deg;
  Texture2D tex;
  bool selected = false;
  bool mouse_bound = false;
  Vector2 relmousepos;
  int fileno;
  std::string filename;
  std::string type;

  json get_state_dict() {
    json jdict;
    jdict["x"] = x;
    jdict["y"] = y;
    jdict["w"] = w;
    jdict["h"] = h;
    jdict["r"] = rotation_deg;
    jdict["fn"] = fileno;
    jdict["file"] = filename;

    return jdict;
  }
};

Rectangle GetPhiMapRectangle(PhiMap im, float scale) {
  Rectangle ret = {(float)im.x, (float)im.y, (float)im.w, (float)im.h};
  if (scale != 0.0) {
    ret.width *= 2 * scale;
    ret.height *= 2 * scale;
  }
  return ret; 
} 

PhiMap LoadPhiMap(const char* filename, int id, float pos=0, std::string folder="new_pngs") {
  fmt::print("Importing : {} to id {}\n", filename, id);
  Image cat = LoadImage(filename);
  ExportImage(cat, fmt::format("{}/{}.png", folder, id).c_str());
  ImageFlipHorizontal(&cat);
  ImageFlipVertical(&cat);
  Texture2D texture = LoadTextureFromImage(cat);      
  UnloadImage(cat);
  // Load 5% app.images (convention) with 100 dpi
  return (PhiMap) { pos*2, -1, texture.width / 100.f * 0.05f, texture.height/100.f*0.05f, 0, texture, false, false, id};
}
PhiMap LoadPhiMapJSON(const char* filename, json d, int id) {
  Image cat = LoadImage(filename);
  ImageFlipHorizontal(&cat);
  ImageFlipVertical(&cat);
  Texture2D texture = LoadTextureFromImage(cat);      
  UnloadImage(cat);
  return (PhiMap) {d["x"], d["y"], d["w"], d["h"], d["r"], texture, false, false, id};
}


enum AppMode { Stitching, Segmenting };
enum CursorMode { Brush, Eraser };
struct ApplicationState {
    AppMode mode = AppMode::Stitching;
    int file_loaded = 0;

    std::vector<PhiMap> images;
    std::vector<PhiMap> backgrounds;
    float * backgrounds_alpha = nullptr;
    float global_scale = 0.5;
    float global_angle = 0.5;
    std::string folder = "new_pngs";
    bool background_edit = 0;
    int  background_cur = 0;

    int  segmentation_base = 0;
    int shown_step = 0;
    bool steps_initialized = false;
    Image steps[5];
    bool use_phi = false;
    bool show_segmentation = true;
    Texture2D steps_tex[5];
    std::size_t* segmentations[3] = {nullptr, nullptr, nullptr};
    Color boundaries_color = RED;
    Color focus_zone_color = PINK;
    bool gui_toggle_active = false;
    int focus_zone_state = 0;
    bool boundaries_dirty = false;
    bool hovering_menus = false;
    float phimaps_alpha = 0.5;
    Rectangle focus_zone = (Rectangle) {1,1, 5, 5};
    Rectangle params_rect;
    struct SegmentPropertiesKM {
      std::size_t key;
      SegmentProperties value;
    };
    struct SegmentSelection {
      std::size_t key;
      bool value;
    };
    
    SegmentPropertiesKM * metadata_labels = nullptr;
    SegmentSelection * selected_labels = nullptr;

    Image drawing_board = {0};
    Image phimap = {0};
    bool drawing_board_active = false;
    Vector2 drawing_board_cursor;
    float drawing_board_cursor_size = 3;
    Texture2D drawing_board_tex = {0};
    int drawing_board_cursor_mode = CursorMode::Brush;

    struct Parameters {
      int kl_strenght = 25;
      int kl_kernel = 9;
      int kl_search_window = 21;
      int qs_kernel_size = 3;
      float qs_ratio = 0.5;
      int qs_max_size = 20;
      float rag_threshold=0.1;
    } params;
};

void
DrawFocusZone(ApplicationState app, Camera2D camera) {
  if (!(app.shown_step == 0 || app.shown_step == 4)) {
    if (app.focus_zone_state) 
      DrawRectangleLinesEx(app.focus_zone, 0.05, app.focus_zone_color);
    else 
      DrawRectangleLinesEx(app.focus_zone, 0.01, app.focus_zone_color);
    
    DrawCircleV((Vector2) {app.focus_zone.x+app.focus_zone.width/2, app.focus_zone.y+app.focus_zone.height/2}, 10.0f / camera.zoom, (Color) {app.focus_zone_color.r, app.focus_zone_color.g, app.focus_zone_color.b, 128});
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      DrawCircleV((Vector2) {app.focus_zone.x, app.focus_zone.y}, 0.1, app.focus_zone_color);
      DrawCircleV((Vector2) {app.focus_zone.x+app.focus_zone.width, app.focus_zone.y}, 0.1, app.focus_zone_color);
      DrawCircleV((Vector2) {app.focus_zone.x+app.focus_zone.width, app.focus_zone.y+app.focus_zone.height}, 0.1, app.focus_zone_color);
      DrawCircleV((Vector2) {app.focus_zone.x, app.focus_zone.y+app.focus_zone.height}, 0.1, app.focus_zone_color);
    }
    
  }
}

std::size_t current_max_label(ApplicationState app, int avoid=-1) {
  Image& start = app.steps[0];
  std::size_t current_max = 0;
  for (int i = 0; i < 3; i++) {
    if (i == avoid) continue;
    current_max = MAXVAL(current_max, maximum_label(app.segmentations[i], start.height*start.width));
  }
  return current_max;
}

void
EnsureWellAllocatedSegments(ApplicationState& app) {
  Image& start = app.steps[0];
  std::size_t length = start.width * start.height;
  for_range(i, 3) {
    if (!app.segmentations[i]) 
    {
      fmt::print("Allocating {} size_t for semgentation {}\n", length, i);
      app.segmentations[i] = (std::size_t*) calloc(length, sizeof(std::size_t));
    }
  }
}

void
LoadAll(char* filename, ApplicationState & app) {
  fmt::print("Loading {}.\n", filename);
  std::ifstream f(filename);
  json data = json::parse(f); 

  if (data.count("global") > 0) {
    app.global_angle = data["global"]["angle"];
    app.global_scale = data["global"]["scale"];
  }
  if (data.count("ui") > 0) {
    auto ui_data = data["ui"];
    if (ui_data.count("boundaries_color") > 0) {
      auto e = ui_data["boundaries_color"];
      app.boundaries_color.r = e[0];
      app.boundaries_color.g = e[1];
      app.boundaries_color.b = e[2];
      app.boundaries_color.a = 255;
    }
  }

  app.images.clear();
  app.folder = data["global"]["folder"];
  for (int i = 0; i < data["patches"].size(); i++) {
      auto d = data["patches"][i];
      app.images.push_back(LoadPhiMapJSON(fmt::format("{}/{}.png",app.folder, i).c_str(), d, i));
  }

  for (auto e : data["backgrounds"]) {
    std::string png_filename = e["file"];
    Image bg = LoadImage(png_filename.c_str());
    Texture2D tex = LoadTextureFromImage(bg);
    app.backgrounds.push_back(PhiMap(e["x"], e["y"], e["w"], e["h"], e["r"], tex, false, false, 0));
    if (e.contains("alpha")) arrput(app.backgrounds_alpha, e["alpha"]);
    else arrput(app.backgrounds_alpha, 0.5);
    app.backgrounds.back().filename = png_filename;
    int count = 0;
    const char ** extension = TextSplit(GetFileName(png_filename.c_str()), '.', &count);
    app.backgrounds.back().type =  fmt::format("{}", extension[1]);
  }
  FILE *read_ptr;
  std::size_t buffer_len;
  char * bin_filename = TextReplace(filename, ".json", ".bin");
  fmt::print("Reading {}.\n\f", bin_filename);
  read_ptr = fopen(bin_filename,"rb");
  if (read_ptr)
  {
    fread(&buffer_len, sizeof(std::size_t), 1, read_ptr);
    uint8_t *buffer = (uint8_t*) malloc(buffer_len);
    fread(buffer, buffer_len-sizeof(std::size_t), 1, read_ptr);
    std::size_t *buffer_as_sizet = (std::size_t*)buffer;
    std::size_t width = buffer_as_sizet[0];
    std::size_t height = buffer_as_sizet[1];
    std::size_t format = buffer_as_sizet[2];
    buffer = (uint8_t*) (buffer_as_sizet+3);
    std::size_t imlength = width*height;
    printf("Loading %lu x %lu images for a total of %lu bytes.\n\f", width, height, buffer_len);
    assert(buffer_len > 0);
    for_range(i, 5) {
      app.steps[i].data = (uint8_t*) malloc(imlength*4*sizeof(uint8_t));
      app.steps[i].height = height;
      app.steps[i].width = width;
      app.steps[i].format = format;
      memcpy(app.steps[i].data, buffer, imlength*4*sizeof(uint8_t));
      buffer += imlength*4*sizeof(uint8_t);
      UnloadTexture(app.steps_tex[i]);
      app.steps_tex[i] = LoadTextureFromImage(app.steps[i]);
    }
    // Segmentations
    for_range(i, 3) {
      app.segmentations[i] = (std::size_t*) malloc(imlength*sizeof(std::size_t));
      memcpy(app.segmentations[i], buffer, imlength*sizeof(std::size_t));
      buffer += imlength*sizeof(std::size_t);
    }
    fclose(read_ptr);
  }
}
void SaveAll(char* filename, ApplicationState& app) {
  fmt::print("Saving {}\n", filename);
  json data;
  data["patches"] = json::array();
  for (int i = 0; i < app.images.size(); i++) {
    data["patches"].push_back(app.images[i].get_state_dict());
  }
  data["backgrounds"] = json::array();
  for (auto & bg : app.backgrounds) {
    data["backgrounds"].push_back(bg.get_state_dict());
  }
  data["global"]["angle"] = app.global_angle;
  data["global"]["scale"] = app.global_scale;
  data["global"]["folder"] = app.folder;

  json bc = json::array();
  bc.push_back(app.boundaries_color.r);
  bc.push_back(app.boundaries_color.g);
  bc.push_back(app.boundaries_color.b);
  data["ui"]["boundaries_color"] = bc;
  
  // Write to disk
  std::ofstream f(filename);
  f << data;
  f.close();

  if(app.steps_initialized) {
    // Save data
    std::size_t buffer_size = 4 * sizeof(std::size_t);
    // Steps images
    std::size_t imlength = app.steps[0].width*app.steps[0].height;
    for_range(i, 5) {
      buffer_size += imlength*4*sizeof(uint8_t);
    }
    // Segmentations
    for_range(i, 3) {
      buffer_size += imlength*sizeof(std::size_t);
    }
    uint8_t *buffer = (uint8_t*) malloc(buffer_size);
    uint8_t *cursor = buffer;
    std::size_t *buffer_as_sizet = (std::size_t*)buffer;
    buffer_as_sizet[0] = buffer_size;
    buffer_as_sizet[1] = app.steps[0].width;
    buffer_as_sizet[2] = app.steps[0].height;
    buffer_as_sizet[3] = app.steps[0].format;
    cursor = (uint8_t*)(buffer_as_sizet + 4);
    for_range(i, 5) {
      memcpy(cursor, app.steps[i].data, imlength*4*sizeof(uint8_t));
      cursor += imlength*4*sizeof(uint8_t);
    }
    // Segmentations
    for_range(i, 3) {
      memcpy(cursor, app.segmentations[i], imlength*sizeof(std::size_t));
      cursor += imlength*sizeof(std::size_t);
    }
    printf("Writing %lu bytes\n", buffer_size);
    char * bin_filename = TextReplace(filename, ".json", ".bin");
    fmt::print("Writing {}.\n\f", bin_filename);
    FILE *write_ptr = NULL;
    write_ptr = fopen(bin_filename,"wb");
    assert(write_ptr);
    fwrite(buffer, sizeof(uint8_t), buffer_size, write_ptr);
    fclose(write_ptr);
  }
}


void FileDialogAction(int action, GuiFileDialogState& fileDialogState, ApplicationState& app) {
    char fileNameToLoad[512] = { 0 };
    switch(action) {
      case 0: {
        if (!IsFileExtension(fileDialogState.fileNameText, ".json")) break;
        fmt::print("Opening {}\n", fileDialogState.fileNameText);
        strcpy(fileNameToLoad, TextFormat("%s/%s", fileDialogState.dirPathText, fileDialogState.fileNameText));
        LoadAll(fileNameToLoad, app);
        app.file_loaded=1;
        break;
      }
      case 1: {
        if (!IsFileExtension(fileDialogState.fileNameText, ".json")) break;
        fmt::print("Saving {}\n", fileDialogState.fileNameText);
        strcpy(fileNameToLoad, TextFormat("%s/%s", fileDialogState.dirPathText, fileDialogState.fileNameText));
        SaveAll(fileNameToLoad, app); 
        break;
      }
      case 2: {
        if (!app.file_loaded) break;
        fmt::print("Loading files from {} : num=", fileDialogState.dirPathText); 
        auto files = LoadDirectoryFiles(fileDialogState.dirPathText);
        fmt::print("{}\n", files.count);
        int cur_size = app.images.size();
        for (int i = 0; i < files.count; i++) {
          fmt::print("{}\n", files.paths[i]);
          app.images.push_back(LoadPhiMap(files.paths[i], cur_size+i, (float)i*app.global_scale, app.folder));
        }
        break;
      }
      case 3:
      {
        strcpy(fileNameToLoad, TextFormat("%s/%s", fileDialogState.dirPathText, fileDialogState.fileNameText));
        fmt::print("Loading background {}", fileNameToLoad); 
        Image bg = LoadImage(fileNameToLoad);
        float aspect_ratio = (float) bg.width / (float) bg.height;

        app.backgrounds.push_back(PhiMap(
            0.f,0.f,
            aspect_ratio * 6.f,6.0f,
          0, LoadTextureFromImage(bg),
          false, false, 0
        ));
        app.backgrounds.back().filename = fileNameToLoad;
        int count = 0;
        const char ** extension = TextSplit(GetFileName(app.backgrounds.back().filename.c_str()), '.', &count);
        app.backgrounds.back().type =  fmt::format("{}", extension[1]);
        arrput(app.backgrounds_alpha, 0.5);
        break;
      }
    };
    fileDialogState.SelectFilePressed = false;
}

void
UpdateBoundariesDisplay(ApplicationState & app, int segmentation, int image) {
  UnloadImage(app.steps[image]);
  UnloadTexture(app.steps_tex[image]);
  if (app.phimap.data ==nullptr) fmt::print("Warning: phimap [p] was not computed\n");
  if (app.show_segmentation) {
    if (app.use_phi &&app.phimap.data !=nullptr)  {
      app.steps[image] = MarkBoundaries(app.phimap, app.segmentations[segmentation], app.boundaries_color);
    }
    else {
      app.steps[image] = MarkBoundaries(app.steps[0], app.segmentations[segmentation], app.boundaries_color);
    }
  }
  else {
    if (app.use_phi && app.phimap.data !=nullptr) {
      app.steps[image] = ImageCopy(app.phimap);
    } else {
      app.steps[image] = ImageCopy(app.steps[0]);
    }
  }
  app.steps_tex[image] = LoadTextureFromImage(app.steps[image]);
}

int main(void)
{
    Image denoised;
    const int screenWidth = 1200;
    const int screenHeight = 720;


    Camera2D camera = { 0 };
    camera.target = (Vector2){ 0, 0 };
    camera.offset = (Vector2){ screenWidth/2.0f, screenHeight/2.0f };
    camera.rotation = 0.0f;
    camera.zoom = 100.0f;
    
    InitWindow(screenWidth, screenHeight, "raylib measurements stitcher");

    GuiFileDialogState fileDialogState = InitGuiFileDialog(GetWorkingDirectory());
    fileDialogState.windowBounds.x = screenWidth / 2;
    fileDialogState.windowBounds.y = screenHeight / 2;
    int action = 0;

    ApplicationState app;
    if (!app.steps_initialized) {
       for (int i = 0; i < 5; i++)
       {
          app.steps[i] = GenImageColor(10,10, (Color){(unsigned char)(50*i), 0,0,255 });
          ImageDrawRectangleLines(&app.steps[i], (Rectangle) {1,1,8,8}, 1, WHITE);
          app.steps_tex[i] = LoadTextureFromImage(app.steps[i]);
       }
       app.steps_initialized = true;
    }

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        auto mp = GetMousePosition();
        auto wmp = GetScreenToWorld2D(mp, camera);
        app.hovering_menus = 
             CheckCollisionPointRec(mp, app.params_rect)
          || mp.y >= screenHeight-20;

        if (IsKeyPressed(KEY_TAB)) {
          app.mode = app.mode == AppMode::Stitching ? AppMode::Segmenting : AppMode::Stitching;
        }
        if (fileDialogState.SelectFilePressed) FileDialogAction(action, fileDialogState, app);
        if (fileDialogState.windowActive) GuiLock();
        if (!fileDialogState.windowActive) {
          if (app.mode == AppMode::Stitching) {
            if (IsKeyPressed(KEY_X)) app.background_edit = !app.background_edit;
            if (IsKeyPressed(KEY_B)) app.background_cur = (app.background_cur + 1) % app.backgrounds.size();
            

            if (!app.background_edit) {
              if (IsKeyDown(KEY_I) && IsKeyDown(KEY_LEFT_SHIFT)) app.global_scale+=0.01;
              else if (IsKeyDown(KEY_I)) app.global_scale+=0.001;
              else if (IsKeyDown(KEY_O) && IsKeyDown(KEY_LEFT_SHIFT)) app.global_scale-=0.01;
              else if (IsKeyDown(KEY_O)) app.global_scale-=0.001;
              else if (IsKeyDown(KEY_K) && IsKeyDown(KEY_LEFT_SHIFT)) app.global_angle+=0.1;
              else if (IsKeyDown(KEY_K)) app.global_angle+=0.01;
              else if (IsKeyDown(KEY_L) && IsKeyDown(KEY_LEFT_SHIFT)) app.global_angle-=0.1;
              else if (IsKeyDown(KEY_L)) app.global_angle-=0.01;
            } else if (app.background_cur < app.backgrounds.size()) {
              PhiMap& target = app.backgrounds[app.background_cur];
              if (IsKeyDown(KEY_I) && IsKeyDown(KEY_LEFT_SHIFT)) target.w +=0.05;
              else if (IsKeyDown(KEY_I)) target.w +=0.005;
              else if (IsKeyDown(KEY_O) && IsKeyDown(KEY_LEFT_SHIFT)) target.w -=0.05;
              else if (IsKeyDown(KEY_O)) target.w -=0.005;
              else if (IsKeyDown(KEY_K) && IsKeyDown(KEY_LEFT_SHIFT)) target.h+=0.05;
              else if (IsKeyDown(KEY_K)) target.h+=0.005;
              else if (IsKeyDown(KEY_L) && IsKeyDown(KEY_LEFT_SHIFT)) target.h-=0.05;
              else if (IsKeyDown(KEY_L)) target.h-=0.005;
              else if (IsKeyDown(KEY_R) && IsKeyDown(KEY_LEFT_SHIFT)) target.rotation_deg+=0.1;
              else if (IsKeyDown(KEY_R)) target.rotation_deg+=0.05;
              else if (IsKeyDown(KEY_T) && IsKeyDown(KEY_LEFT_SHIFT)) target.rotation_deg-=0.1;
              else if (IsKeyDown(KEY_T)) target.rotation_deg-=0.05;
            }
            if (IsKeyPressed(KEY_P) && app.segmentation_base < app.backgrounds.size()) {
              // Locate the background used as base for segmentation
              PhiMap & bg = app.backgrounds[app.segmentation_base];
              float x0=bg.x, y0=bg.y, w0=bg.w, h0 = bg.h;
              int mul = 1;
              if (IsKeyDown(KEY_LEFT_SHIFT)) mul = 4;
              int px = mul * bg.tex.width, py=mul * bg.tex.height;
              // Generate and save the phimap
              RenderTexture2D rt = LoadRenderTexture(px, py);
              BeginTextureMode(rt);
              ClearBackground(WHITE);
              for(int i = 0; i < app.images.size(); i++) {
                PhiMap im = app.images[i];
                Rectangle dest = {(float) px*(im.x-x0)/w0, (float) py*(im.y-y0)/h0, (float) px*im.w*2*app.global_scale/w0, (float) py*im.h*2*app.global_scale/h0};
                DrawTexturePro(im.tex, {0.f, 0.f, (float) im.tex.width, (float) im.tex.height}, dest, {0.f,0.f}, app.global_angle, WHITE);
              }
              EndTextureMode();
              if (app.phimap.data == nullptr)
                app.phimap = LoadImageFromTexture(rt.texture);
              else
                UnloadImage(app.phimap);
                app.phimap = LoadImageFromTexture(rt.texture);
              ImageFlipVertical(&app.phimap);
              ExportImage(app.phimap, "phimap_new.png");
              UnloadRenderTexture(rt);
            }
            if (IsKeyPressed(KEY_G) && app.segmentation_base < app.backgrounds.size()) {
              // Locate the background used as base for segmentation
              PhiMap & bg = app.backgrounds[app.segmentation_base];
              float x0=bg.x, y0=bg.y, w0=bg.w, h0 = bg.h;
              int mul = 1;
              if (IsKeyDown(KEY_LEFT_SHIFT)) mul = 4;
              int px = mul * bg.tex.width, py=mul * bg.tex.height;
              // Generate and save the phimap
              RenderTexture2D rt = LoadRenderTexture(px, py);
              BeginTextureMode(rt);
              ClearBackground(BLACK);
              Color bgcolors[3];
              bgcolors[0] = {255,0,0,255};
              bgcolors[1] = {0,255,0,128};
              bgcolors[2] = {0,0,255, 85};
              for(int i = 0; i < app.backgrounds.size(); i++) {
                PhiMap im = app.backgrounds[i];
                Rectangle dest = {(float) px*(im.x-x0)/w0, (float) py*(im.y-y0)/h0, (float) px*im.w*2*app.global_scale/w0, (float) py*im.h*2*app.global_scale/h0};
                Image bg1 = LoadImageFromTexture(im.tex);
                ImageColorGrayscale(&bg1);
                Texture2D bg1tex = LoadTextureFromImage(bg1);
                DrawTexturePro(bg1tex, {0.f, 0.f, (float) im.tex.width, (float) im.tex.height}, dest, {0.f,0.f}, im.rotation_deg, bgcolors[i]);
              }
              EndTextureMode();
              if (app.phimap.data == nullptr)
                app.phimap = LoadImageFromTexture(rt.texture);
              else {
                UnloadImage(app.phimap);
                app.phimap = LoadImageFromTexture(rt.texture);
              }
              unsigned char maxval[3] = {0,0,0};
              for(int i = 0; i < app.phimap.width*app.phimap.height; i++) {
                for (int j = 0; j < 3; j++) {
                  int cur = ((unsigned char*) app.phimap.data)[i*4+j];
                  maxval[j] = cur > maxval[j] ? cur : maxval[j];
                }
              }
              printf("Maxvals %d %d %d\n", maxval[0], maxval[1], maxval[2]);
              for(int i = 0; i < app.phimap.width*app.phimap.height; i++) {
                for (int j = 0; j < 3; j++) {
                  unsigned char& cur = ((unsigned char*) app.phimap.data)[i*4+j];
                  cur = round((255 * cur) / (float) maxval[j]) ;
                }
                unsigned char& cur = ((unsigned char*) app.phimap.data)[i*4+3];
                cur = 255;
              }
              printf("Image format: %d\n", app.phimap.format);
              ImageFlipVertical(&app.phimap);
              ExportImage(app.phimap, "compound_background.png");
              UnloadRenderTexture(rt);
            }
            // Objects under left-clicked cursor are mouse-bound.
            // We also register the position relative to the cursor.
            // This is more stable for moving objects around than using mouse delta
            if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
            {
             
              for(int i = 0; i < app.images.size(); i++) {
                auto im = app.images[i];
                if(CheckCollisionPointRec(wmp, GetPhiMapRectangle(im, app.global_scale))) {
                  app.images[i].selected = !app.images[i].selected;
                  app.images[i].mouse_bound = true;
                  app.images[i].relmousepos.x = mp.x - GetWorldToScreen2D((Vector2) {(float) im.x, (float) im.y}, camera).x;
                  app.images[i].relmousepos.y = mp.y - GetWorldToScreen2D((Vector2) {(float) im.x, (float) im.y}, camera).y;
                  break;
                }
              }
            } // Mouse-binding

            // Releasing lmb releases all mouse-bound items. 
            if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT))
            {
              for(int i = 0; i < app.images.size(); i++) {
                app.images[i].mouse_bound = false;
              }
            } // Binding release

            // Link mouse-bound objects positions to mouse position
            for(int i = 0; i < app.images.size(); i++) {
                auto im = app.images[i];
                if(im.mouse_bound)
                {
                  auto mp = GetMousePosition();
                  auto op = GetWorldToScreen2D((Vector2) {(float) im.x, (float) im.y}, camera);
                  op.x = mp.x - im.relmousepos.x;
                  op.y = mp.y - im.relmousepos.y;
                  op = GetScreenToWorld2D(op, camera);
                  app.images[i].x = op.x;
                  app.images[i].y = op.y;
                }
            } // Bound-move
            if (app.background_edit && app.background_cur < app.backgrounds.size() && IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !app.hovering_menus)
            {
              PhiMap& target = app.backgrounds[app.background_cur];
              auto dmp = GetMouseDelta();
              auto tp = GetWorldToScreen2D((Vector2) {(float) target.x, (float) target.y}, camera);
              tp.x += dmp.x;
              tp.y += dmp.y;
              target.x = GetScreenToWorld2D(tp, camera).x;
              target.y = GetScreenToWorld2D(tp, camera).y;
            } // Background-move
          } // AppMode::Stitching
          else if (app.mode == AppMode::Segmenting) {
            if (IsKeyPressed(KEY_V)) {
              app.use_phi = !app.use_phi;
              app.boundaries_dirty = true;
            }
            if (IsKeyPressed(KEY_C)) {
              app.show_segmentation = !app.show_segmentation;
              app.boundaries_dirty = true;
            }
            if (IsKeyPressed(KEY_X)) hmfree(app.selected_labels);
            if (app.shown_step != 0 && app.shown_step != 4) {
              if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && IsKeyDown(KEY_LEFT_SHIFT)) {
                if (CheckCollisionPointCircle((Vector2) {wmp.x, wmp.y}, (Vector2) {app.focus_zone.x, app.focus_zone.y}, 0.1))
                  app.focus_zone_state = 0b0001;
                else if (CheckCollisionPointCircle((Vector2) {wmp.x, wmp.y}, (Vector2) {app.focus_zone.x+app.focus_zone.width, app.focus_zone.y}, 0.1))
                  app.focus_zone_state = 0b0010;
                else if (CheckCollisionPointCircle((Vector2) {wmp.x, wmp.y}, (Vector2) {app.focus_zone.x, app.focus_zone.y + app.focus_zone.height}, 0.1))
                  app.focus_zone_state = 0b0100;
                else if (CheckCollisionPointCircle((Vector2) {wmp.x, wmp.y}, (Vector2) {app.focus_zone.x+app.focus_zone.width, app.focus_zone.y+app.focus_zone.height}, 0.1))
                  app.focus_zone_state = 0b1000;
              } else if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && CheckCollisionPointCircle((Vector2) {wmp.x, wmp.y}, (Vector2) {app.focus_zone.x+app.focus_zone.width/2, app.focus_zone.y+app.focus_zone.height/2}, 0.1)) {
                  app.focus_zone_state = 0b1111;
              } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
                app.focus_zone_state = 0;
              }
            } else if (app.shown_step == 4 && app.segmentation_base < app.backgrounds.size()) {
              // When we look at manual rag
              PhiMap& bg = app.backgrounds[app.segmentation_base];
              if(IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && (IsKeyDown(KEY_LEFT_CONTROL) || !app.drawing_board_active) && CheckCollisionPointRec(wmp, (Rectangle) {bg.x, bg.y, bg.w, bg.h}) && !app.hovering_menus) {
                int x = round((wmp.x - bg.x) / bg.w * bg.tex.width);
                int y = round((wmp.y - bg.y) / bg.h * bg.tex.height);

                std::size_t id = app.segmentations[2][y * bg.tex.width + x];
                if (hmgeti(app.selected_labels, id) == -1)
                  hmput(app.selected_labels, id, true);
                else
                  hmdel(app.selected_labels, id);

                if (hmgeti(app.metadata_labels, id) != -1) {
                  SegmentProperties& seg = hmget(app.metadata_labels, id);
                } else {
                  SegmentProperties lab = ComputeSegmentProperties(app.segmentations[2], bg.tex.width, bg.tex.height, id);
                  hmput(app.metadata_labels, id, lab);
                }
              }
              if (app.drawing_board_active && !IsKeyDown(KEY_LEFT_CONTROL)) {
                if(!app.drawing_board.data) app.drawing_board= GenImageColor(bg.tex.width, bg.tex.height, {0,0,0,0});
                if (CheckCollisionPointRec(wmp, (Rectangle) {bg.x, bg.y, bg.w, bg.h})&&!app.hovering_menus) {
                  int x = round((wmp.x - bg.x) / bg.w * bg.tex.width);
                  int y = round((wmp.y - bg.y) / bg.h * bg.tex.height);
                  int brush_size = bg.tex.width*app.drawing_board_cursor_size/ 4.0;
                  #define for_interval(X, MIN, MAX) for(int X = (MIN); X < (MAX); (X)++)
                  int nsel = hmlen(app.selected_labels);
                  Color brush_color = app.drawing_board_cursor_mode == CursorMode::Brush ? YELLOW : NOCOLOR;
                  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                    for_interval(i_, x-brush_size/2, x+brush_size/2) {
                      for_interval(j_, y-brush_size/2, y+brush_size/2) {
                        if (sqrtf((i_-x)*(i_-x)+(j_-y)*(j_-y)) < brush_size/2) {
                          if (nsel==0) {
                            // If no segment is selected, draw
                            ((Color*)app.drawing_board.data)[j_ * bg.tex.width+ i_] = brush_color;
                          } else if (hmgeti(app.selected_labels, app.segmentations[2][j_ * bg.tex.width+ i_]) != -1) {
                            // If one is selected, we only draw on selected pixels.
                            ((Color*)app.drawing_board.data)[j_ * bg.tex.width+ i_] = brush_color;
                          }
                        }
                      }
                    }
                    if (app.drawing_board_tex.id > 0) UnloadTexture(app.drawing_board_tex);
                    app.drawing_board_tex = LoadTextureFromImage(app.drawing_board);
                  }

                }
              }
            }

            if (app.focus_zone_state) {
              if (app.focus_zone_state == 0b1111) {
                app.focus_zone.x = wmp.x - app.focus_zone.width / 2;
                app.focus_zone.y = wmp.y - app.focus_zone.height / 2;
              }
              else if (app.focus_zone_state & 0b0001) { 
                Rectangle old = app.focus_zone;
                app.focus_zone.x = wmp.x;
                app.focus_zone.y = wmp.y;
                app.focus_zone.width -= app.focus_zone.x - old.x;
                app.focus_zone.height -= app.focus_zone.y - old.y;
              } else if (app.focus_zone_state & 0b0010) { 
                Rectangle old = app.focus_zone;
                app.focus_zone.y = wmp.y;
                app.focus_zone.width = wmp.x - old.x;
                app.focus_zone.height -= app.focus_zone.y - old.y;
              } else if (app.focus_zone_state & 0b0100) { 
                Rectangle old = app.focus_zone;
                app.focus_zone.x = wmp.x;
                app.focus_zone.width -= app.focus_zone.x - old.x;
                app.focus_zone.height = wmp.y - old.y;
              } else if (app.focus_zone_state & 0b1000) { 
                Rectangle old = app.focus_zone;
                app.focus_zone.width = wmp.x - old.x;
                app.focus_zone.height = wmp.y - old.y;
              }
            }
            app.focus_zone.width = fmax(app.focus_zone.width, 0.1);
            app.focus_zone.height = fmax(app.focus_zone.height, 0.1);

            // Segmentation processing controls
            if (app.segmentation_base < app.backgrounds.size()) {
              if (IsKeyPressed(KEY_T)) {
                PhiMap & bg = app.backgrounds[app.segmentation_base];
                if (IsKeyDown(KEY_LEFT_SHIFT)) {
                  UnloadImage(app.steps[0]);
                  UnloadImage(app.steps[1]);
                  app.steps[0] = LoadImageFromTexture(bg.tex);
                  app.steps[1] = ImageCopy(app.steps[0]);
                  opencv_nlmeans_denoising(app.steps[0], app.steps[1], app.params.kl_strenght, app.params.kl_kernel, app.params.kl_search_window);
                  UnloadTexture(app.steps_tex[0]);
                  UnloadTexture(app.steps_tex[1]);
                  app.steps_tex[0] = LoadTextureFromImage(app.steps[0]);
                  app.steps_tex[1] = LoadTextureFromImage(app.steps[1]);
                } else {
                  float x0 = bg.x, y0 = bg.y, w0 = bg.w,h0 = bg.h;
                  Rectangle fi = app.focus_zone;
                  Rectangle focus_pixels = { round((fi.x - x0) / w0 * bg.tex.width), round((fi.y - y0) / h0 * bg.tex.height), round(fi.width/w0*bg.tex.width), round(fi.height/h0*bg.tex.height)};
                  Image crop = ImageFromImage(app.steps[0], focus_pixels);
                  Image dcrop = ImageCopy(crop);
                  opencv_nlmeans_denoising(crop, dcrop, app.params.kl_strenght, app.params.kl_kernel, app.params.kl_search_window);
                  DrawImageOnImage(app.steps[1], dcrop, focus_pixels);
                  UnloadTexture(app.steps_tex[1]);
                  app.steps_tex[1] = LoadTextureFromImage(app.steps[1]);
                }
              }
              if (IsKeyPressed(KEY_Y)) {
                EnsureWellAllocatedSegments(app);
                PhiMap & bg = app.backgrounds[app.segmentation_base];
                Image& start = app.steps[1];
                int length = start.height*start.width;
                if (IsKeyDown(KEY_LEFT_SHIFT)) {
                  quickshift(start, app.params.qs_kernel_size, app.params.qs_max_size, app.segmentations[0], app.params.qs_ratio);
                  auto offset = current_max_label(app, 0);
                  relabel_sequential(app.segmentations[0], length, offset);
                  relabel_sequential_global(app.segmentations, length);
                  UpdateBoundariesDisplay(app, 0, 2);
                } else {
                  float x0 = bg.x, y0 = bg.y, w0 = bg.w,h0 = bg.h;
                  Rectangle fi = app.focus_zone;
                  Rectangle focus_pixels = { round((fi.x - x0) / w0 * start.width), round((fi.y - y0) / h0 * start.height), round(fi.width/w0*start.width), round(fi.height/h0*start.height)};
                  Image crop = ImageFromImage(app.steps[0], focus_pixels);
                  std::size_t *cropseg = (std::size_t*) malloc(focus_pixels.width*focus_pixels.height*sizeof(std::size_t));
                  quickshift(crop, app.params.qs_kernel_size, app.params.qs_max_size, cropseg, app.params.qs_ratio);
                  auto offset = current_max_label(app);
                  relabel_sequential(cropseg, crop.width*crop.height, offset);
                  DrawImageOnImageL(app.segmentations[0], cropseg, focus_pixels, start.width, start.height, 0);
                  UnloadImage(crop);
                  free(cropseg);

                }
                relabel_sequential_global(app.segmentations, length);
                UpdateBoundariesDisplay(app, 0, 2);
              }
              if (IsKeyPressed(KEY_U) ) {
                EnsureWellAllocatedSegments(app);
                Image& start = app.steps[1];
                if (IsKeyDown(KEY_LEFT_SHIFT)) {
                  int length = start.height*start.width;
                  memcpy(app.segmentations[1], app.segmentations[0], length*sizeof(std::size_t));
                  std::size_t *labels = app.segmentations[1];
                  std::size_t num_components = maximum_label(labels, length) + 1;
                  rag r = rag_create(num_components);
                  rag_adjacency_matrix(r, labels, start.width, start.height);

                  float *im_float = uint8_to_float((uint8_t*)start.data, length, 4, 3);
                  rag_color_distance_matrix(r, im_float, labels, start.width, start.height);
                  free(im_float);

                  rag_merge(r, app.params.rag_threshold);
                  rag_relabel(r, labels, length);
                } else {
                  PhiMap & bg = app.backgrounds[app.segmentation_base];
                  float x0 = bg.x, y0 = bg.y, w0 = bg.w,h0 = bg.h;
                  Rectangle fi = app.focus_zone;
                  Rectangle focus_pixels = { round((fi.x - x0) / w0 * start.width), round((fi.y - y0) / h0 * start.height), round(fi.width/w0*start.width), round(fi.height/h0*start.height)};
                  std::size_t * crop = ImageFromImageL(app.segmentations[0], focus_pixels, start.width, start.height);
                  Image imcrop = ImageFromImage(app.steps[0], focus_pixels);
                  std::size_t width = focus_pixels.width;
                  std::size_t height = focus_pixels.height;
                  std::size_t length = height*width;
                  std::size_t num_components = maximum_label(crop, length) + 1;
                  rag r = rag_create(num_components);
                  rag_adjacency_matrix(r, crop, width, height);

                  float *im_float = uint8_to_float((uint8_t*)imcrop.data, length, 4, 3);
                  rag_color_distance_matrix(r, im_float, crop, width, height);
                  free(im_float);

                  rag_merge(r, app.params.rag_threshold);
                  rag_relabel(r, crop, length);

                  //auto offset = current_max_label(app);
                  
                  DrawImageOnImageL(app.segmentations[1], crop, focus_pixels, start.width, start.height, 0);
                  free(crop);
                  UnloadImage(imcrop);
                }
                hmfree(app.metadata_labels);
                hmfree(app.selected_labels);
                UpdateBoundariesDisplay(app, 1, 3);
              }
              //if (IsKeyPressed(KEY_O)) {
              //  EnsureWellAllocatedSegments(app);
              //  Image& start = app.steps[2];
              //  int length = start.height*start.width;
              //  UnloadImage(app.steps[3]);
              //  UnloadTexture(app.steps_tex[3]);
              //  memcpy(app.segmentations[1], app.segmentations[0], length*sizeof(std::size_t));
              //  app.steps[3] = ImageCopy(app.steps[2]);
              //  app.steps_tex[3] = LoadTextureFromImage(app.steps[3]);
              //  hmfree(app.metadata_labels);
              //  hmfree(app.selected_labels);
              //}
              if (IsKeyPressed(KEY_I) ) {
                EnsureWellAllocatedSegments(app);
                Image& start = app.steps[0];
                if (IsKeyDown(KEY_LEFT_SHIFT)) {
                  int length = start.height*start.width;
                  memcpy(app.segmentations[2], app.segmentations[1], length*sizeof(std::size_t));
                } else {
                  PhiMap & bg = app.backgrounds[app.segmentation_base];
                  float x0 = bg.x, y0 = bg.y, w0 = bg.w,h0 = bg.h;
                  Rectangle fi = app.focus_zone;
                  Rectangle focus_pixels = { round((fi.x - x0) / w0 * start.width), round((fi.y - y0) / h0 * start.height), round(fi.width/w0*start.width), round(fi.height/h0*start.height)};
                  std::size_t * crop = ImageFromImageL(app.segmentations[1], focus_pixels, start.width, start.height);
                  std::size_t width = focus_pixels.width;
                  std::size_t height = focus_pixels.height;
                  std::size_t length = height*width;
                  DrawImageOnImageL(app.segmentations[2], crop, focus_pixels, start.width, start.height, 0);
                  free(crop);
                }
                hmfree(app.metadata_labels);
                hmfree(app.selected_labels);
                UpdateBoundariesDisplay(app, 2, 4);
              }
              if (IsKeyPressed(KEY_B) && IsKeyDown(KEY_LEFT_SHIFT)) {
                EnsureWellAllocatedSegments(app);
                Image& start = app.steps[0];
                auto current_max = current_max_label(app);
                for_range(i, start.height*start.width) {
                  if (((Color*)app.drawing_board.data)[i].r == YELLOW.r) {
                    app.segmentations[2][i] = current_max + 1;
                    ((Color*)app.drawing_board.data)[i] = {0,0,0,0};
                  }
                }
                if (app.drawing_board_tex.id > 0) UnloadTexture(app.drawing_board_tex);
                app.drawing_board_tex = LoadTextureFromImage(app.drawing_board);
                app.boundaries_dirty = true;
                hmfree(app.metadata_labels);
                hmfree(app.selected_labels);
              }
              if (IsKeyPressed(KEY_J) && IsKeyDown(KEY_LEFT_SHIFT) && hmlen(app.selected_labels) > 1) {
                Image& start = app.steps[0];
                std::size_t master_id = app.selected_labels[0].key;
                for_range(i, start.height*start.width) {
                  if (hmgeti(app.selected_labels, app.segmentations[2][i]) != -1) {
                    app.segmentations[2][i] = master_id;
                  }
                }
                app.boundaries_dirty = true;
                hmfree(app.metadata_labels);
                hmfree(app.selected_labels);
              }
              if (app.boundaries_dirty) {
                UpdateBoundariesDisplay(app, 0, 2);
                UpdateBoundariesDisplay(app, 1, 3);
                UpdateBoundariesDisplay(app, 2, 4);
                app.boundaries_dirty = false;
              }
            } // Has a segmentation base image

          } // AppMode::Segmenting

          /**
           * Commong to all modes
          */
          // Zoom on canvas
          if (IsKeyDown(KEY_LEFT_CONTROL) && GetMouseWheelMove() != 0.0) {
            camera.zoom += GetMouseWheelMove();
            camera.zoom = fmax(100, camera.zoom);
            if (GetMouseWheelMove() > 0)
            {
              // Move towards the cursor while zooming
              auto alpha = fmin(fabs(GetMouseWheelMove() / 50), 1.0);
              camera.target.x = (1-alpha) * camera.target.x + (alpha) * wmp.x;
              camera.target.y = (1-alpha) * camera.target.y + (alpha) * wmp.y;
            }
          } // Canvas Zoom

          // Link canvas position to mouse position
          if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT))
          {
            auto dmp = GetMouseDelta();
            auto tp = GetWorldToScreen2D(camera.target, camera);
            tp.x -= dmp.x;
            tp.y -= dmp.y;
            camera.target = GetScreenToWorld2D(tp, camera);
          } // Canvas Move
        }
        /** ----------- *
         *    Drawing
         *  ----------- */
        
        BeginDrawing();
          BeginMode2D(camera);
            ClearBackground(RAYWHITE);
            if(app.mode == AppMode::Stitching) {
              Color color = WHITE;
              if (app.background_edit && app.background_cur < app.backgrounds.size()) {
                color = {255,255,255, 128};
                for (int i = 0; i < app.backgrounds.size(); i++) {
                  PhiMap& target = app.backgrounds[i];
                  DrawTexturePro(target.tex, {0.f, 0.f, (float) target.tex.width, (float) target.tex.height}, {target.x, target.y, target.w, target.h}, {0.f,0.f}, target.rotation_deg, {255,255,255, (uint8_t) (255*app.backgrounds_alpha[i])});
                }
                PhiMap& target = app.backgrounds[app.background_cur];
                DrawRectangleLinesEx((Rectangle) {target.x, target.y, target.w, target.h}, 0.05, RED);
              }
              else if (app.background_cur < app.backgrounds.size()) {
                PhiMap& bg = app.backgrounds[app.background_cur];
                DrawTexturePro(bg.tex, {0.f, 0.f, (float) bg.tex.width, (float) bg.tex.height}, {bg.x, bg.y, bg.w, bg.h}, {0.f,0.f}, bg.rotation_deg, color);
              }
              if (!app.background_edit) {
                for(int i = 0; i < app.images.size(); i++) {
                  PhiMap im = app.images[i];
                  Rectangle dest = {(float) im.x, (float) im.y, (float) im.w*2*app.global_scale, (float) im.h*2*app.global_scale};
                  DrawTexturePro(im.tex, {0.f, 0.f, (float) im.tex.width, (float) im.tex.height}, dest, {0.f,0.f}, app.global_angle, (Color) {255,255,255, (uint8_t)(255*app.phimaps_alpha)});
                }
              }
            } // AppMode::Stitching
            else if (app.mode == AppMode::Segmenting) {
              Texture2D& tex = app.steps_tex[app.shown_step];
              Rectangle dest = { 0, 0, 6,6};
              if (app.segmentation_base < app.backgrounds.size()) {
                PhiMap& bg = app.backgrounds[app.segmentation_base];
                dest.x = bg.x;
                dest.y = bg.y;
                dest.width = bg.w;
                dest.height = bg.h;
                DrawTexturePro(tex, (Rectangle) {0,0,(float)tex.width, (float)tex.height}, dest, (Vector2) {0,0}, 0, WHITE );
                for_range(i, hmlen(app.selected_labels)) {
                  auto id = app.selected_labels[i].key;
                  auto e = hmget(app.metadata_labels, id);
                  Rectangle bbox= e.bbox;
                  Vector2 wcentroid = { (e.centroid.x / bg.tex.width) * bg.w + bg.x, (e.centroid.y / bg.tex.height) * bg.h + bg.y};
                  Rectangle bbox_on_canvas = (Rectangle) {
                    (bbox.x / (float) tex.width) * bg.w + bg.x, 
                    (bbox.y / (float) tex.height) * bg.h + bg.y, 
                    bbox.width / (float) tex.width * bg.w,
                    bbox.height / (float) tex.height * bg.h,
                    };
                  DrawTexturePro(e.blob_tex, (Rectangle) {0,0, (float) e.blob_tex.width, (float) e.blob_tex.height}, bbox_on_canvas, (Vector2) {0,0}, 0, WHITE);
                  //DrawRectangleLinesEx(bbox_on_canvas, 0.02, BLUE);
                }
                if (app.drawing_board_active) {
                  Texture2D& tt = app.drawing_board_tex;
                  DrawTexturePro(app.drawing_board_tex, (Rectangle){0,0,(float)tt.width, (float)tt.height}, dest, (Vector2) {0.f,0.f}, 0.f, {255,255,255,128});
                  if (CheckCollisionPointRec(wmp, (Rectangle) {bg.x, bg.y, bg.w, bg.h})) {
                    DrawCircleV({wmp.x, wmp.y}, app.drawing_board_cursor_size, {255,200,200,100});
                    int x = round((wmp.x - bg.x) / bg.w * bg.tex.width);
                    int y = round((wmp.y - bg.y) / bg.h * bg.tex.height);
                  }
                }
              } // Has background
              
              DrawFocusZone(app, camera);
            } // AppMode::Segmenting

          DrawGuidingLines(BLACK, camera); 
          
          EndMode2D();

          /*
           * Start GUI Definition
           */

          if (GuiButton((Rectangle){ 0, 0, 140, 30 }, GuiIconText(ICON_FILE_OPEN, "Open Project"))) {
            fileDialogState.saveFileMode = false;
            fileDialogState.dirPathEditMode = false;
            fileDialogState.windowActive = true;
            action = 0;
          }
          if (GuiButton((Rectangle){ 140, 0, 140, 30 }, GuiIconText(ICON_FILE_SAVE, "Save Project"))) {
            fileDialogState.saveFileMode = true;
            fileDialogState.dirPathEditMode = false;
            fileDialogState.windowActive = true;
            action = 1;
          }
          if (GuiButton((Rectangle){ 280, 0, 140, 30 }, GuiIconText(ICON_FILE_ADD, "Add Pictures"))) {
            fileDialogState.saveFileMode = false;
            fileDialogState.dirPathEditMode = true;
            fileDialogState.windowActive = true;
            action = 2;
          }
          if (GuiButton((Rectangle){ 420, 0, 140, 30 }, GuiIconText(ICON_FILE_OPEN, "Add Background"))) {
            fileDialogState.saveFileMode = false;
            fileDialogState.dirPathEditMode = false;
            fileDialogState.windowActive = true;
            action = 3;
          }
          app.mode = (AppMode) GuiComboBox((Rectangle){ screenWidth-140, 0, 140, 30 }, "stitching;segmenting", app.mode);
          if (app.background_cur < app.backgrounds.size()) DrawText(GetFileNameWithoutExt(app.backgrounds[app.background_cur].filename.c_str()), 5*screenWidth/8,7*screenHeight/8, 20, RED);
          Rectangle wp = {100,100, 500, 500};
          wp.x = 0;
          wp.y = screenHeight-20;
          app.background_edit = GuiToggle((Rectangle) {wp.x,wp.y,100,20}, "Background edit", app.background_edit);
          std::string cb_data = "";
          if (app.backgrounds.size() > 0) {
            cb_data = app.backgrounds.front().type.c_str();
            for(int i = 1; i < app.backgrounds.size(); i++) {
              cb_data+= ";";
              cb_data+= app.backgrounds[i].type.c_str();
            }
          }
          app.background_cur = GuiComboBox((Rectangle) {wp.x + 100, wp.y, 250, 20}, cb_data.c_str(), app.background_cur);
          if (app.mode == AppMode::Stitching) app.segmentation_base = GuiComboBox((Rectangle) {wp.x+350, wp.y, 250, 20}, cb_data.c_str(), app.segmentation_base);
          if (app.mode == AppMode::Segmenting && app.segmentation_base < app.backgrounds.size()) 
            GuiLabel((Rectangle) {wp.x+350, wp.y, 250, 20}, app.backgrounds[app.segmentation_base].type.c_str());
          else if(app.mode == AppMode::Segmenting)
            GuiLabelButton((Rectangle) {wp.x+350, wp.y, 250, 20}, GuiIconText(ICON_EMPTYBOX, "NO BACKGROUND TO SEGMENT"));

          if (app.mode == AppMode::Segmenting)  app.shown_step = GuiComboBox((Rectangle) {wp.x+600, wp.y, 250, 20}, "base;denosied;quickshift;ragged;manual", app.shown_step );
          if (app.mode == AppMode::Segmenting)  app.drawing_board_active = GuiToggle((Rectangle) {wp.x+850, wp.y, 50, 20}, "freedraw", app.drawing_board_active );
          float x_start = screenWidth - 280;
          float x_sliders = x_start + 80;
          float y_start = screenHeight-300;
          float w_sliders = 150;
          app.params_rect = (Rectangle) {x_start, y_start-RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT, 300,300+RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT};
          GuiPanel((Rectangle) {x_start, y_start-RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT, 300,300+RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT}, "Parameters");
          if (app.mode == AppMode::Segmenting) {
            GuiGroupBox((Rectangle) {x_start+10, y_start+10, w_sliders+110, 80}, "Denosing [t]");
            #define PARAM_SLIDER(VAR, YSHIFT, TITLE, MIN, MAX) VAR = GuiSlider((Rectangle) {x_sliders, (YSHIFT), w_sliders, 20}, TITLE, fmt::format("{}", VAR).c_str(), VAR, MIN, MAX);
            PARAM_SLIDER(app.params.kl_strenght,      y_start+20, "KL str.",     0.0f, 50.0f);
            PARAM_SLIDER(app.params.kl_search_window, y_start+40, "KL search.",  0.0f, 50.0f);
            PARAM_SLIDER(app.params.kl_kernel,        y_start+60, "KL kern.",    0.0f, 50.0f);
            
            GuiGroupBox((Rectangle) {x_start+10, y_start+100, w_sliders+110, 65}, "Quickshift [y]");
            PARAM_SLIDER(app.params.qs_kernel_size, y_start+105, "QS kern.",     0.0f, 50.0f);
            PARAM_SLIDER(app.params.qs_max_size,    y_start+125, "QS max size",  0.0f, 50.0f);
            PARAM_SLIDER(app.params.qs_ratio,       y_start+145, "QS ratio",     0.0f, 1.0f );

            GuiGroupBox((Rectangle) {x_start+10, y_start+180, w_sliders+110, 30}, "Region Adjacency [u]");
            PARAM_SLIDER(app.params.rag_threshold, y_start+190, "RAG thr.", 0.0f, 0.2f);
            
            bool old = app.gui_toggle_active;
            app.gui_toggle_active = GuiToggle((Rectangle) {x_sliders, y_start+215, 100,10}, "Boundaries Color", app.gui_toggle_active);
            if (app.gui_toggle_active) {
              app.boundaries_color = GuiColorPicker((Rectangle) {screenWidth-500, screenHeight-200, 120,120}, "", app.boundaries_color);
            } else if (old != app.gui_toggle_active) {
              app.boundaries_dirty = true;
            }

            GuiGroupBox((Rectangle) {x_start+10, y_start+230, w_sliders+110, 30}, "Drawing []");
            PARAM_SLIDER(app.drawing_board_cursor_size, y_start+240, "Brush size.", 0.0001f, 0.2f);
            app.drawing_board_cursor_mode = GuiComboBox((Rectangle) {x_start+10, y_start+260, 100, 20}, "brush;eraser", app.drawing_board_cursor_mode);
          } // AppMode::Segmenting
          else if (app.mode == AppMode::Stitching) {
            GuiGroupBox((Rectangle) {x_start+10, y_start+10, w_sliders+110, 40}, "PhiMaps [t]");
            app.phimaps_alpha = GuiColorBarAlpha((Rectangle) {x_sliders, y_start+20, w_sliders,20}, "Phi alpha", app.phimaps_alpha);
            GuiGroupBox((Rectangle) {x_start+10, y_start+60, w_sliders+110, (float) arrlen(app.backgrounds_alpha)*20+20}, "Backgrounds [t]");
            for_range(i, arrlen(app.backgrounds_alpha)) {
              app.backgrounds_alpha[i] = GuiColorBarAlpha((Rectangle) {x_sliders, y_start+70+20*i, w_sliders,15}, app.backgrounds[i].type.c_str(), app.backgrounds_alpha[i]);
            }
          }
          for (int i  = 0; i < hmlen(app.selected_labels); i++)
            DrawText(fmt::format("Selected {}", app.selected_labels[i].key).c_str(), GetScreenWidth()-100, 40 + i, 12, BLACK);
          GuiUnlock();
          GuiFileDialog(&fileDialogState);

        EndDrawing();
    } // Main loop
    CloseWindow();                // Close window and OpenGL context
    return 0;
}
