REST API reference
==================


.. raw:: html

    <!-- Attempts to embed the docs inside sphinx were not sucessfull so far
    <script>
    $(document).ready(function () {

        var inner_doc = document.getElementById("swagger-docs").contentDocument
        var inner_width = inner_doc.body.getBoundingClientRect().height;

        $('#swagger-docs-container').height(inner_width);
        $('#swagger-docs').height(inner_width);

        $('#sidenav', inner_doc).hide();
        $('#content', inner_doc).css('margin-left', 0);
        $('.container-fluid', inner_doc).css('width', $('.rst-content').width());
        $('.container-fluid', inner_doc).css('padding', 0);


    });

    </script>


    <div id='swagger-docs-container'>
    <object id='swagger-docs'data="./swagger_docs/index.html" style="display:block; overflow:hidden; position: absolute; height: 100%; width: 100%" frameborder="0" scrolling="no" ></object>
    <div>
    -->

The REST API reference can be found `here <./openapi-docs/index.html>`_.

FreeDiscovery server also exposes the `OpenAPI specification <https://www.openapis.org/about>`_ for the REST API at `/openapi-specs.json`, and the automatic documentation with Swagger UI at `/swagger-ui.html`.
