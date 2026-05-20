//! SVG document builder — fluent API for constructing SVG documents
//! with shapes, paths, text, groups, gradients, patterns, transforms,
//! animation, and styling.
//!
//! Pure-Rust replacement for D3 SVG, Snap.svg, and SVG.js.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Core types ──────────────────────────────────────────────────

/// An SVG attribute key-value pair.
#[derive(Debug, Clone)]
struct Attr {
    key: String,
    value: String,
}

/// An SVG node in the document tree.
#[derive(Debug, Clone)]
pub struct SvgNode {
    tag: String,
    attrs: Vec<Attr>,
    children: Vec<SvgNode>,
    text_content: Option<String>,
    self_closing: bool,
}

impl SvgNode {
    fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            attrs: Vec::new(),
            children: Vec::new(),
            text_content: None,
            self_closing: false,
        }
    }

    fn new_self_closing(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            attrs: Vec::new(),
            children: Vec::new(),
            text_content: None,
            self_closing: true,
        }
    }

    /// Set an attribute on this node.
    pub fn attr(mut self, key: &str, value: &str) -> Self {
        self.attrs.push(Attr {
            key: key.to_string(),
            value: value.to_string(),
        });
        self
    }

    /// Set an f64 attribute.
    pub fn attr_f(mut self, key: &str, value: f64) -> Self {
        self.attrs.push(Attr {
            key: key.to_string(),
            value: format!("{value}"),
        });
        self
    }

    /// Set the text content.
    pub fn text(mut self, content: &str) -> Self {
        self.text_content = Some(content.to_string());
        self.self_closing = false;
        self
    }

    /// Add a child node.
    pub fn child(mut self, node: SvgNode) -> Self {
        self.self_closing = false;
        self.children.push(node);
        self
    }

    /// Add multiple children.
    pub fn children(mut self, nodes: Vec<SvgNode>) -> Self {
        self.self_closing = false;
        self.children.extend(nodes);
        self
    }

    /// Set the `id` attribute.
    pub fn id(self, id: &str) -> Self {
        self.attr("id", id)
    }

    /// Set the `class` attribute.
    pub fn class(self, cls: &str) -> Self {
        self.attr("class", cls)
    }

    /// Set fill color.
    pub fn fill(self, color: &str) -> Self {
        self.attr("fill", color)
    }

    /// Set stroke color.
    pub fn stroke(self, color: &str) -> Self {
        self.attr("stroke", color)
    }

    /// Set stroke width.
    pub fn stroke_width(self, width: f64) -> Self {
        self.attr_f("stroke-width", width)
    }

    /// Set opacity.
    pub fn opacity(self, val: f64) -> Self {
        self.attr_f("opacity", val)
    }

    /// Set a style attribute.
    pub fn style(self, css: &str) -> Self {
        self.attr("style", css)
    }

    /// Set transform.
    pub fn transform(self, t: &str) -> Self {
        self.attr("transform", t)
    }

    /// Render to SVG string.
    pub fn to_string_pretty(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write_to(&mut out, indent, 0);
        out
    }

    fn write_to(&self, out: &mut String, indent_size: usize, depth: usize) {
        let indent = " ".repeat(indent_size * depth);

        out.push_str(&indent);
        out.push('<');
        out.push_str(&self.tag);

        for attr in &self.attrs {
            let _ = write!(out, " {}=\"{}\"", attr.key, xml_escape(&attr.value));
        }

        if self.self_closing && self.children.is_empty() && self.text_content.is_none() {
            out.push_str(" />\n");
            return;
        }

        out.push('>');

        if let Some(text) = &self.text_content {
            out.push_str(&xml_escape(text));
        }

        if !self.children.is_empty() {
            out.push('\n');
            for child in &self.children {
                child.write_to(out, indent_size, depth + 1);
            }
            out.push_str(&indent);
        }

        out.push_str("</");
        out.push_str(&self.tag);
        out.push_str(">\n");
    }
}

impl fmt::Display for SvgNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_pretty(2))
    }
}

// ── SVG document builder ────────────────────────────────────────

/// Builder for creating complete SVG documents.
pub struct SvgBuilder {
    width: Option<f64>,
    height: Option<f64>,
    view_box: Option<(f64, f64, f64, f64)>,
    xmlns: bool,
    elements: Vec<SvgNode>,
    defs: Vec<SvgNode>,
    styles: Vec<String>,
}

impl SvgBuilder {
    pub fn new() -> Self {
        Self {
            width: None,
            height: None,
            view_box: None,
            xmlns: true,
            elements: Vec::new(),
            defs: Vec::new(),
            styles: Vec::new(),
        }
    }

    /// Set the width and height.
    pub fn size(mut self, width: f64, height: f64) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set the viewBox.
    pub fn view_box(mut self, min_x: f64, min_y: f64, width: f64, height: f64) -> Self {
        self.view_box = Some((min_x, min_y, width, height));
        self
    }

    /// Whether to include xmlns attribute (default true).
    pub fn xmlns(mut self, include: bool) -> Self {
        self.xmlns = include;
        self
    }

    /// Add a shape element.
    pub fn add(mut self, element: SvgNode) -> Self {
        self.elements.push(element);
        self
    }

    /// Add a definition (gradient, pattern, etc.).
    pub fn def(mut self, element: SvgNode) -> Self {
        self.defs.push(element);
        self
    }

    /// Add a CSS style block.
    pub fn style(mut self, css: &str) -> Self {
        self.styles.push(css.to_string());
        self
    }

    /// Build the SVG document string.
    pub fn build(&self) -> String {
        let mut root = SvgNode::new("svg");

        if self.xmlns {
            root = root.attr("xmlns", "http://www.w3.org/2000/svg");
        }

        if let Some(w) = self.width {
            root = root.attr_f("width", w);
        }
        if let Some(h) = self.height {
            root = root.attr_f("height", h);
        }
        if let Some((min_x, min_y, w, h)) = self.view_box {
            root = root.attr("viewBox", &format!("{min_x} {min_y} {w} {h}"));
        }

        // Add <style> if present.
        if !self.styles.is_empty() {
            let css = self.styles.join("\n");
            let style_node = SvgNode::new("style")
                .attr("type", "text/css")
                .text(&css);
            root = root.child(style_node);
        }

        // Add <defs> if present.
        if !self.defs.is_empty() {
            let mut defs_node = SvgNode::new("defs");
            for d in &self.defs {
                defs_node = defs_node.child(d.clone());
            }
            root = root.child(defs_node);
        }

        // Add elements.
        for elem in &self.elements {
            root = root.child(elem.clone());
        }

        root.to_string_pretty(2)
    }
}

impl Default for SvgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Shape constructors ──────────────────────────────────────────

/// Create a `<rect>` element.
pub fn rect(x: f64, y: f64, width: f64, height: f64) -> SvgNode {
    SvgNode::new_self_closing("rect")
        .attr_f("x", x)
        .attr_f("y", y)
        .attr_f("width", width)
        .attr_f("height", height)
}

/// Create a `<circle>` element.
pub fn circle(cx: f64, cy: f64, r: f64) -> SvgNode {
    SvgNode::new_self_closing("circle")
        .attr_f("cx", cx)
        .attr_f("cy", cy)
        .attr_f("r", r)
}

/// Create an `<ellipse>` element.
pub fn ellipse(cx: f64, cy: f64, rx: f64, ry: f64) -> SvgNode {
    SvgNode::new_self_closing("ellipse")
        .attr_f("cx", cx)
        .attr_f("cy", cy)
        .attr_f("rx", rx)
        .attr_f("ry", ry)
}

/// Create a `<line>` element.
pub fn line(x1: f64, y1: f64, x2: f64, y2: f64) -> SvgNode {
    SvgNode::new_self_closing("line")
        .attr_f("x1", x1)
        .attr_f("y1", y1)
        .attr_f("x2", x2)
        .attr_f("y2", y2)
}

/// Create a `<polyline>` element from a series of points.
pub fn polyline(points: &[(f64, f64)]) -> SvgNode {
    let pts: Vec<String> = points.iter().map(|(x, y)| format!("{x},{y}")).collect();
    SvgNode::new_self_closing("polyline").attr("points", &pts.join(" "))
}

/// Create a `<polygon>` element from a series of points.
pub fn polygon(points: &[(f64, f64)]) -> SvgNode {
    let pts: Vec<String> = points.iter().map(|(x, y)| format!("{x},{y}")).collect();
    SvgNode::new_self_closing("polygon").attr("points", &pts.join(" "))
}

/// Create a `<path>` element.
pub fn path(d: &str) -> SvgNode {
    SvgNode::new_self_closing("path").attr("d", d)
}

/// Create a `<text>` element.
pub fn text(x: f64, y: f64, content: &str) -> SvgNode {
    SvgNode::new("text")
        .attr_f("x", x)
        .attr_f("y", y)
        .text(content)
}

/// Create a `<tspan>` element.
pub fn tspan(content: &str) -> SvgNode {
    SvgNode::new("tspan").text(content)
}

/// Create a `<g>` (group) element.
pub fn group() -> SvgNode {
    SvgNode::new("g")
}

/// Create a `<use>` element referencing an id.
pub fn use_ref(href: &str) -> SvgNode {
    SvgNode::new_self_closing("use").attr("href", &format!("#{href}"))
}

/// Create an `<image>` element.
pub fn image(href: &str, x: f64, y: f64, width: f64, height: f64) -> SvgNode {
    SvgNode::new_self_closing("image")
        .attr("href", href)
        .attr_f("x", x)
        .attr_f("y", y)
        .attr_f("width", width)
        .attr_f("height", height)
}

/// Create a `<clipPath>` element.
pub fn clip_path(id: &str) -> SvgNode {
    SvgNode::new("clipPath").id(id)
}

/// Create a `<mask>` element.
pub fn mask(id: &str) -> SvgNode {
    SvgNode::new("mask").id(id)
}

// ── Gradient constructors ───────────────────────────────────────

/// Create a `<linearGradient>` definition.
pub fn linear_gradient(id: &str, stops: &[(f64, &str)]) -> SvgNode {
    let mut grad = SvgNode::new("linearGradient").id(id);
    for (offset, color) in stops {
        let pct = format!("{}%", offset * 100.0);
        grad = grad.child(
            SvgNode::new_self_closing("stop")
                .attr("offset", &pct)
                .attr("stop-color", color),
        );
    }
    grad
}

/// Create a `<radialGradient>` definition.
pub fn radial_gradient(id: &str, stops: &[(f64, &str)]) -> SvgNode {
    let mut grad = SvgNode::new("radialGradient").id(id);
    for (offset, color) in stops {
        let pct = format!("{}%", offset * 100.0);
        grad = grad.child(
            SvgNode::new_self_closing("stop")
                .attr("offset", &pct)
                .attr("stop-color", color),
        );
    }
    grad
}

/// Create a `<pattern>` definition.
pub fn pattern(id: &str, width: f64, height: f64) -> SvgNode {
    SvgNode::new("pattern")
        .id(id)
        .attr_f("width", width)
        .attr_f("height", height)
        .attr("patternUnits", "userSpaceOnUse")
}

// ── Animation constructors ──────────────────────────────────────

/// Create an `<animate>` element.
pub fn animate(attribute: &str, from: &str, to: &str, dur: &str) -> SvgNode {
    SvgNode::new_self_closing("animate")
        .attr("attributeName", attribute)
        .attr("from", from)
        .attr("to", to)
        .attr("dur", dur)
}

/// Create an `<animateTransform>` element.
pub fn animate_transform(
    transform_type: &str,
    from: &str,
    to: &str,
    dur: &str,
) -> SvgNode {
    SvgNode::new_self_closing("animateTransform")
        .attr("attributeName", "transform")
        .attr("type", transform_type)
        .attr("from", from)
        .attr("to", to)
        .attr("dur", dur)
}

/// Create an `<animateMotion>` element.
pub fn animate_motion(path_d: &str, dur: &str) -> SvgNode {
    SvgNode::new_self_closing("animateMotion")
        .attr("dur", dur)
        .attr("path", path_d)
}

/// Create a `<set>` element.
pub fn set_attr(attribute: &str, to: &str) -> SvgNode {
    SvgNode::new_self_closing("set")
        .attr("attributeName", attribute)
        .attr("to", to)
}

// ── Transform helpers ───────────────────────────────────────────

/// Build a translate transform string.
pub fn translate(x: f64, y: f64) -> String {
    format!("translate({x}, {y})")
}

/// Build a rotate transform string.
pub fn rotate(angle: f64) -> String {
    format!("rotate({angle})")
}

/// Build a rotate transform around a center point.
pub fn rotate_around(angle: f64, cx: f64, cy: f64) -> String {
    format!("rotate({angle}, {cx}, {cy})")
}

/// Build a scale transform string.
pub fn scale(sx: f64, sy: f64) -> String {
    format!("scale({sx}, {sy})")
}

/// Build a skewX transform string.
pub fn skew_x(angle: f64) -> String {
    format!("skewX({angle})")
}

/// Build a skewY transform string.
pub fn skew_y(angle: f64) -> String {
    format!("skewY({angle})")
}

// ── Helpers ─────────────────────────────────────────────────────

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_svg() {
        let svg = SvgBuilder::new().size(100.0, 100.0).build();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("width=\"100\""));
        assert!(svg.contains("height=\"100\""));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_viewbox() {
        let svg = SvgBuilder::new()
            .view_box(0.0, 0.0, 200.0, 200.0)
            .build();
        assert!(svg.contains("viewBox=\"0 0 200 200\""));
    }

    #[test]
    fn test_rect() {
        let svg = SvgBuilder::new()
            .size(100.0, 100.0)
            .add(rect(10.0, 20.0, 50.0, 30.0).fill("red"))
            .build();
        assert!(svg.contains("<rect"));
        assert!(svg.contains("x=\"10\""));
        assert!(svg.contains("fill=\"red\""));
    }

    #[test]
    fn test_circle_element() {
        let svg = SvgBuilder::new()
            .add(circle(50.0, 50.0, 25.0).stroke("blue").stroke_width(2.0))
            .build();
        assert!(svg.contains("<circle"));
        assert!(svg.contains("cx=\"50\""));
        assert!(svg.contains("r=\"25\""));
    }

    #[test]
    fn test_line_element() {
        let svg = SvgBuilder::new()
            .add(line(0.0, 0.0, 100.0, 100.0).stroke("black"))
            .build();
        assert!(svg.contains("<line"));
        assert!(svg.contains("x1=\"0\""));
        assert!(svg.contains("x2=\"100\""));
    }

    #[test]
    fn test_path_element() {
        let svg = SvgBuilder::new()
            .add(path("M 0 0 L 100 100 Z").fill("none").stroke("green"))
            .build();
        assert!(svg.contains("d=\"M 0 0 L 100 100 Z\""));
    }

    #[test]
    fn test_text_element() {
        let svg = SvgBuilder::new()
            .add(text(10.0, 20.0, "Hello SVG").attr("font-size", "16"))
            .build();
        assert!(svg.contains("<text"));
        assert!(svg.contains("Hello SVG"));
        assert!(svg.contains("</text>"));
    }

    #[test]
    fn test_group() {
        let svg = SvgBuilder::new()
            .add(
                group()
                    .transform(&translate(10.0, 20.0))
                    .child(circle(0.0, 0.0, 5.0))
                    .child(rect(0.0, 0.0, 10.0, 10.0)),
            )
            .build();
        assert!(svg.contains("<g"));
        assert!(svg.contains("translate(10, 20)"));
        assert!(svg.contains("<circle"));
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn test_linear_gradient() {
        let svg = SvgBuilder::new()
            .def(linear_gradient("grad1", &[(0.0, "red"), (1.0, "blue")]))
            .add(rect(0.0, 0.0, 100.0, 100.0).fill("url(#grad1)"))
            .build();
        assert!(svg.contains("<defs>"));
        assert!(svg.contains("<linearGradient"));
        assert!(svg.contains("id=\"grad1\""));
        assert!(svg.contains("<stop"));
    }

    #[test]
    fn test_radial_gradient() {
        let svg = SvgBuilder::new()
            .def(radial_gradient("rg1", &[(0.0, "white"), (1.0, "black")]))
            .build();
        assert!(svg.contains("<radialGradient"));
        assert!(svg.contains("id=\"rg1\""));
    }

    #[test]
    fn test_pattern_def() {
        let svg = SvgBuilder::new()
            .def(
                pattern("dots", 10.0, 10.0).child(circle(5.0, 5.0, 2.0).fill("gray")),
            )
            .build();
        assert!(svg.contains("<pattern"));
        assert!(svg.contains("id=\"dots\""));
    }

    #[test]
    fn test_use_ref_element() {
        let svg = SvgBuilder::new()
            .def(circle(0.0, 0.0, 10.0).id("dot"))
            .add(use_ref("dot").attr_f("x", 50.0))
            .build();
        assert!(svg.contains("href=\"#dot\""));
    }

    #[test]
    fn test_animation() {
        let animated = circle(50.0, 50.0, 10.0).child(animate("r", "10", "30", "2s"));
        let svg = SvgBuilder::new().add(animated).build();
        assert!(svg.contains("<animate"));
        assert!(svg.contains("attributeName=\"r\""));
    }

    #[test]
    fn test_animate_transform_element() {
        let el = rect(0.0, 0.0, 20.0, 20.0)
            .child(animate_transform("rotate", "0 10 10", "360 10 10", "3s"));
        let svg = SvgBuilder::new().add(el).build();
        assert!(svg.contains("<animateTransform"));
        assert!(svg.contains("type=\"rotate\""));
    }

    #[test]
    fn test_style_block() {
        let svg = SvgBuilder::new()
            .style(".cls { fill: red; }")
            .add(rect(0.0, 0.0, 10.0, 10.0).class("cls"))
            .build();
        assert!(svg.contains("<style"));
        assert!(svg.contains(".cls { fill: red; }"));
    }

    #[test]
    fn test_xmlns() {
        let with = SvgBuilder::new().xmlns(true).build();
        assert!(with.contains("xmlns=\"http://www.w3.org/2000/svg\""));

        let without = SvgBuilder::new().xmlns(false).build();
        assert!(!without.contains("xmlns"));
    }

    #[test]
    fn test_polygon_element() {
        let svg = SvgBuilder::new()
            .add(polygon(&[(0.0, 0.0), (50.0, 100.0), (100.0, 0.0)]).fill("yellow"))
            .build();
        assert!(svg.contains("<polygon"));
        assert!(svg.contains("points="));
    }

    #[test]
    fn test_polyline_element() {
        let svg = SvgBuilder::new()
            .add(polyline(&[(0.0, 0.0), (25.0, 50.0), (50.0, 0.0)]).fill("none"))
            .build();
        assert!(svg.contains("<polyline"));
    }

    #[test]
    fn test_ellipse_element() {
        let svg = SvgBuilder::new()
            .add(ellipse(100.0, 50.0, 80.0, 30.0))
            .build();
        assert!(svg.contains("<ellipse"));
        assert!(svg.contains("rx=\"80\""));
        assert!(svg.contains("ry=\"30\""));
    }

    #[test]
    fn test_transform_helpers() {
        assert_eq!(translate(10.0, 20.0), "translate(10, 20)");
        assert_eq!(rotate(45.0), "rotate(45)");
        assert_eq!(scale(2.0, 3.0), "scale(2, 3)");
        assert_eq!(skew_x(30.0), "skewX(30)");
    }

    #[test]
    fn test_id_and_class() {
        let node = rect(0.0, 0.0, 10.0, 10.0).id("myRect").class("highlight");
        let s = node.to_string_pretty(2);
        assert!(s.contains("id=\"myRect\""));
        assert!(s.contains("class=\"highlight\""));
    }

    #[test]
    fn test_opacity() {
        let node = circle(50.0, 50.0, 25.0).opacity(0.5);
        let s = node.to_string_pretty(2);
        assert!(s.contains("opacity=\"0.5\""));
    }

    #[test]
    fn test_xml_escaping() {
        let node = text(0.0, 0.0, "A & B < C");
        let s = node.to_string_pretty(2);
        assert!(s.contains("A &amp; B &lt; C"));
    }

    #[test]
    fn test_nested_groups() {
        let svg = SvgBuilder::new()
            .add(
                group()
                    .id("outer")
                    .child(group().id("inner").child(circle(0.0, 0.0, 5.0))),
            )
            .build();
        assert!(svg.contains("id=\"outer\""));
        assert!(svg.contains("id=\"inner\""));
    }

    #[test]
    fn test_clip_path_element() {
        let cp = clip_path("clip1").child(rect(0.0, 0.0, 50.0, 50.0));
        let svg = SvgBuilder::new()
            .def(cp)
            .add(circle(50.0, 50.0, 40.0).attr("clip-path", "url(#clip1)"))
            .build();
        assert!(svg.contains("<clipPath"));
        assert!(svg.contains("id=\"clip1\""));
    }
}
