#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "raylib.h"

typedef struct {
    float error;
    float metric;
} Sample;

typedef struct {
    Sample *items;
    size_t count;
} SampleBuffer;

static SampleBuffer sample_buffer_new(void) {
    SampleBuffer buffer;
    buffer.items = NULL;
    buffer.count = 0;
    return buffer;
}

static void sample_buffer_free(SampleBuffer *buffer) {
    if (buffer->items) {
        free(buffer->items);
        buffer->items = NULL;
    }
    buffer->count = 0;
}

static bool sample_buffer_push(SampleBuffer *buffer, Sample sample) {
    size_t new_count = buffer->count + 1;
    Sample *next = realloc(buffer->items, new_count * sizeof(Sample));
    if (!next) {
        return false;
    }
    next[buffer->count] = sample;
    buffer->items = next;
    buffer->count = new_count;
    return true;
}

static SampleBuffer load_samples_from_csv(const char *path, bool *loaded) {
    SampleBuffer buffer = sample_buffer_new();
    *loaded = false;

    FILE *handle = fopen(path, "r");
    if (!handle) {
        return buffer;
    }

    char line[512];
    while (fgets(line, sizeof(line), handle)) {
        char *ptr = line;
        while (*ptr == ' ' || *ptr == '\t') {
            ptr++;
        }
        if (*ptr == '\0' || *ptr == '\n' || *ptr == '#') {
            continue;
        }

        char *end_ptr = NULL;
        errno = 0;
        float error = strtof(ptr, &end_ptr);
        if (errno != 0 || end_ptr == ptr) {
            continue;
        }

        while (*end_ptr == ',' || *end_ptr == ' ' || *end_ptr == '\t') {
            end_ptr++;
        }
        errno = 0;
        float metric = strtof(end_ptr, NULL);
        if (errno != 0) {
            continue;
        }

        if (!sample_buffer_push(&buffer, (Sample){error, metric})) {
            break;
        }
    }

    fclose(handle);
    if (buffer.count > 0) {
        *loaded = true;
    }

    return buffer;
}

static SampleBuffer generate_samples(size_t count) {
    SampleBuffer buffer = sample_buffer_new();
    for (size_t i = 0; i < count; ++i) {
        float t = (float)i / (float)count;
        float error = 0.5f + 0.45f * sinf(6.28318f * t) + 0.1f * sinf(23.0f * t);
        float metric = 0.5f + 0.4f * cosf(4.0f * t + 0.5f);
        sample_buffer_push(&buffer, (Sample){error, metric});
    }
    return buffer;
}

static void compute_bounds(const SampleBuffer *buffer, float *min_error, float *max_error,
                           float *min_metric, float *max_metric) {
    if (buffer->count == 0) {
        *min_error = 0.0f;
        *max_error = 1.0f;
        *min_metric = 0.0f;
        *max_metric = 1.0f;
        return;
    }

    *min_error = buffer->items[0].error;
    *max_error = buffer->items[0].error;
    *min_metric = buffer->items[0].metric;
    *max_metric = buffer->items[0].metric;

    for (size_t i = 1; i < buffer->count; ++i) {
        float err = buffer->items[i].error;
        float met = buffer->items[i].metric;
        if (err < *min_error) {
            *min_error = err;
        }
        if (err > *max_error) {
            *max_error = err;
        }
        if (met < *min_metric) {
            *min_metric = met;
        }
        if (met > *max_metric) {
            *max_metric = met;
        }
    }
}

static float normalize(float value, float min_value, float max_value) {
    float span = max_value - min_value;
    if (span <= 0.0f) {
        return 0.5f;
    }
    return (value - min_value) / span;
}

int main(int argc, char **argv) {
    const char *default_path = "generated/scope_snapshot.csv";
    const char *path = (argc > 1) ? argv[1] : default_path;

    bool loaded = false;
    SampleBuffer buffer = load_samples_from_csv(path, &loaded);
    if (!loaded) {
        sample_buffer_free(&buffer);
        buffer = generate_samples(240);
    }

    const int screen_width = 1200;
    const int screen_height = 720;
    InitWindow(screen_width, screen_height, "Perceptrome Scope Visualizer (raylib)");
    SetTargetFPS(60);

    size_t view_start = 0;
    size_t view_count = buffer.count > 0 ? buffer.count : 1;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_R)) {
            view_start = 0;
            view_count = buffer.count;
        }
        if (IsKeyPressed(KEY_LEFT)) {
            if (view_start > 0) {
                view_start--;
            }
        }
        if (IsKeyPressed(KEY_RIGHT)) {
            if (view_start + view_count < buffer.count) {
                view_start++;
            }
        }
        if (IsKeyPressed(KEY_UP)) {
            if (view_count > 10) {
                view_count -= 10;
            }
        }
        if (IsKeyPressed(KEY_DOWN)) {
            if (view_count + 10 <= buffer.count) {
                view_count += 10;
            }
        }

        float min_error, max_error, min_metric, max_metric;
        compute_bounds(&buffer, &min_error, &max_error, &min_metric, &max_metric);

        BeginDrawing();
        ClearBackground((Color){20, 22, 30, 255});

        const int padding = 60;
        Rectangle plot = {padding, padding, screen_width - 2 * padding, screen_height - 2 * padding};

        DrawRectangleLinesEx(plot, 1.0f, (Color){80, 82, 90, 255});

        DrawText("error", plot.x + 10, plot.y + 10, 18, (Color){220, 120, 120, 255});
        DrawText("metric", plot.x + 10, plot.y + 32, 18, (Color){120, 180, 230, 255});

        if (buffer.count > 1) {
            size_t end = view_start + view_count;
            if (end > buffer.count) {
                end = buffer.count;
            }
            float step = plot.width / (float)(end - view_start - 1);
            for (size_t i = view_start; i + 1 < end; ++i) {
                float x0 = plot.x + (float)(i - view_start) * step;
                float x1 = plot.x + (float)(i - view_start + 1) * step;

                float err0 = normalize(buffer.items[i].error, min_error, max_error);
                float err1 = normalize(buffer.items[i + 1].error, min_error, max_error);
                float met0 = normalize(buffer.items[i].metric, min_metric, max_metric);
                float met1 = normalize(buffer.items[i + 1].metric, min_metric, max_metric);

                float y0 = plot.y + plot.height - err0 * plot.height;
                float y1 = plot.y + plot.height - err1 * plot.height;
                float y2 = plot.y + plot.height - met0 * plot.height;
                float y3 = plot.y + plot.height - met1 * plot.height;

                DrawLineEx((Vector2){x0, y0}, (Vector2){x1, y1}, 2.0f, (Color){220, 120, 120, 255});
                DrawLineEx((Vector2){x0, y2}, (Vector2){x1, y3}, 2.0f, (Color){120, 180, 230, 255});
            }
        }

        char status[256];
        snprintf(status, sizeof(status),
                 "samples=%zu  view=%zu..%zu  source=%s", buffer.count, view_start,
                 (view_start + view_count > buffer.count) ? buffer.count : (view_start + view_count),
                 loaded ? path : "synthetic");
        DrawText(status, padding, screen_height - padding + 10, 18, (Color){200, 200, 200, 255});
        DrawText("Controls: Left/Right scroll  Up/Down zoom  R reset", padding, screen_height - padding + 32, 18,
                 (Color){150, 150, 150, 255});

        EndDrawing();
    }

    CloseWindow();
    sample_buffer_free(&buffer);
    return 0;
}
